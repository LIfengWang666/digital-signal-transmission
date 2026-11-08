"""Join measured Stage 1 evidence and draw the sampling decision summary.

This reporting step reads existing CSV; it performs no new FFT, sampling,
reconstruction, or stochastic experiment.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parent


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a named CSV without modifying the source measurements.

    Args:
        path: Existing UTF-8 measurement CSV.

    Returns:
        Rows keyed by the CSV header names, with string values.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Save reporting rows with a header in a stable column order.

    Args:
        path: Destination CSV within this reporting directory.
        rows: Nonempty rows sharing the same keys and column order.

    Returns:
        None. Write the rows to disk.
    """
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Check cross-stage consistency and present the measured decision.

    Returns:
        None. Save joined CSV, provenance hashes, and PNG/SVG figures;
        raise if source measurements contradict one another.
    """
    results = ROOT / "results"
    figures = ROOT / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    sources = [
        STAGE / "r11/results/summary.csv",
        STAGE / "r11/results/components.csv",
        STAGE / "r11/results/spectrum.csv",
        STAGE / "r12/results/summary.csv",
        STAGE / "r12/results/rate_comparison.csv",
        STAGE / "r13/results/summary.csv",
        STAGE / "r13/results/comparison.csv",
    ]
    loaded = [read_csv(path) for path in sources]
    reference, components, spectrum, sampling, rates, alias, compared = loaded
    reference = {r["metric"]: float(r["value"]) for r in reference}
    sampling = {r["metric"]: float(r["value"]) for r in sampling}
    alias = {r["metric"]: float(r["value"]) for r in alias}
    band = reference["band_edge_hz"]
    selected_fs = sampling["selected_sample_rate_hz"]
    selected = next(
        row for row in rates if float(row["sample_rate_hz"]) == selected_fs
    )
    baseline = next(
        row for row in compared if float(row["sample_rate_hz"]) == selected_fs
    )
    under = next(
        row
        for row in compared
        if float(row["sample_rate_hz"]) == alias["under_fs_hz"]
    )
    checks = [
        {
            "check": "band edge agrees across components and stages",
            "passed": band
            == max(float(r["frequency_hz"]) for r in components)
            == sampling["measured_band_edge_hz"]
            == alias["band_edge_hz"],
        },
        {
            "check": "all measured candidates share reference duration",
            "passed": all(
                float(r["duration_s"]) == reference["record_duration_s"]
                for r in rates + compared
            ),
        },
        {
            "check": "chosen rate and sample count agree across stages",
            "passed": selected_fs == alias["baseline_fs_hz"]
            and int(selected["sample_count"])
            == int(baseline["sample_count"])
            == sampling["selected_sample_count"],
        },
        {
            "check": "chosen full-record MSE agrees across stages",
            "passed": float(selected["mse"])
            == float(baseline["mse"])
            == sampling["selected_mse"]
            == alias["baseline_mse"],
        },
        {
            "check": "undersampling error agrees with summary",
            "passed": float(under["mse"]) == alias["under_mse"],
        },
        {
            "check": "spectrum retains complete native frequency grid",
            "passed": len(spectrum) == reference["sample_count"] // 2 + 1
            and float(spectrum[0]["frequency_hz"]) == 0
            and float(spectrum[-1]["frequency_hz"])
            == reference["sample_rate_hz"] / 2,
        },
    ]
    write_csv(results / "checks.csv", checks)
    if not all(row["passed"] for row in checks):
        raise ValueError("Inconsistent Stage 1 reporting sources; see checks.")

    joined = []
    for row in sorted(
        rates + [under], key=lambda r: float(r["sample_rate_hz"])
    ):
        fs = float(row["sample_rate_hz"])
        joined.append(
            {
                "sample_rate_hz": fs,
                "sample_count": int(row["sample_count"]),
                "duration_s": float(row["duration_s"]),
                "nyquist_frequency_hz": float(row["nyquist_frequency_hz"]),
                "measured_band_edge_hz": band,
                "guard_hz": float(row["nyquist_frequency_hz"]) - band,
                "strict_nyquist": fs > 2 * band,
                "selected": fs == selected_fs,
                "raw_payload_ratio_to_selected": fs / selected_fs,
                "mse_npy_units_squared": float(row["mse"]),
                "measurement_source": (
                    "r13/results/comparison.csv"
                    if fs == alias["under_fs_hz"]
                    else "r12/results/rate_comparison.csv"
                ),
            }
        )
    write_csv(results / "decision_summary.csv", joined)
    (results / "reporting_sources.json").write_text(
        json.dumps(
            {
                "role": "Reporting joins only; original measurements retained",
                "sources": [
                    {
                        "path_relative_to_stage1": path.relative_to(
                            STAGE
                        ).as_posix(),
                        "sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    }
                    for path in sources
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    frequency = np.array([float(r["frequency_hz"]) for r in spectrum])
    level = np.array([float(r["rectangular_relative_db"]) for r in spectrum])
    # Match the other Stage 1 figures in standalone and full-pipeline runs.
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.8), layout="constrained")
    ax = axes[0]
    ax.plot(frequency / 1000, np.maximum(level, -140), color="#254b65", lw=0.8)
    for row in components:
        f = float(row["frequency_hz"]) / 1000
        value = float(row["relative_db"])
        ax.plot(f, value, "o", color="#254b65", ms=3)
        ax.annotate(
            f"{f:g}",
            (f, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.axvline(
        band / 1000,
        color="#a43b42",
        ls=":",
        label=f"Measured edge {band / 1000:g} kHz",
    )
    ax.axvline(
        selected_fs / 2000,
        color="#24746a",
        ls="--",
        label=f"Chosen Nyquist frequency {selected_fs / 2000:g} kHz",
    )
    ax.axhline(reference["threshold_db"], color="0.4", ls="--", lw=0.8)
    ax.text(
        23.5,
        reference["threshold_db"] + 4,
        "Detection threshold",
        ha="right",
        fontsize=8,
    )
    ax.set(
        xlim=(0, frequency[-1] / 1000),
        ylim=(-145, 20),
        xlabel="Frequency (kHz)",
        ylabel="Amplitude (dB rel. max)",
    )
    ax.set_title(
        f"(a) Measured reference: all {int(reference['sample_count']):,} "
        "samples, rectangular window",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(alpha=0.15)

    ax = axes[1]
    colors = []
    labels = []
    for row in joined:
        fs = row["sample_rate_hz"]
        labels.append(
            f"{fs / 1000:g}" + (" (chosen)" if row["selected"] else "")
        )
        if row["selected"]:
            colors.append("#24746a")
        elif not row["strict_nyquist"]:
            colors.append("#a43b42")
        else:
            colors.append("#718392")
    values = [r["nyquist_frequency_hz"] / 1000 for r in joined]
    ax.barh(labels, values, color=colors, height=0.62)
    ax.invert_yaxis()
    ax.axvline(
        band / 1000,
        color="black",
        ls=":",
        lw=1,
        label=f"Measured edge $f_B={band / 1000:g}$ kHz",
    )
    for i, row in enumerate(joined):
        suffix = " (boundary)" if row["guard_hz"] == 0 else ""
        ax.text(
            values[i] + 0.25,
            i,
            f"{values[i]:g}{suffix}",
            va="center",
            fontsize=8,
        )
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set(
        xlim=(0, 27),
        xlabel="Nyquist frequency $f_N=f_s/2$ (kHz)",
        ylabel="Sampling rate $f_s$ (kHz)",
    )
    ax.set_title(
        "(b) Strict sufficiency requires each bar to extend beyond "
        f"{band / 1000:g} kHz",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.15)
    for suffix in ["png", "svg"]:
        fig.savefig(
            figures / f"sampling_decision.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("R1.4: joined five measured rates; six consistency checks passed.")


if __name__ == "__main__":
    main()
