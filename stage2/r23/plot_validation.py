"""Plot R2.3 validation from saved CSV, without fresh random draws."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent


def read_rows(name: str) -> list[dict[str, str]]:
    """Read one result table.

    Args:
        name: Filename relative to the saved results directory.

    Returns:
        CSV rows keyed by their original headers.
    """
    with (ROOT / "results" / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    """Draw quality/severity and predetermined waveform/boundary examples.

    Returns:
        None. Save two publication-scale PNG/SVG figures.
    """
    plt.rcdefaults()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    rows = read_rows("summary.csv")
    rows = [row for row in rows if int(row["rate_hz"]) == 48000]
    order = [0, 1, 5, 2, 3, 4, 6, 7]
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    theory = read_rows("theory.csv")
    first = next(row for row in theory if int(row["rate_hz"]) == 48000)
    distortion = float(first["clean_mse"])
    gain = float(first["finite_noise_gain"])
    untreated_gain = float(first["no_filter_noise_gain"])
    sample_power = float(first["expected_no_filter_mse"]) / (
        untreated_gain * 10 ** (-float(first["target_snr_db"]) / 10)
    )
    verification = json.loads(
        (ROOT.parents[1] / "metadata/r23_verification.json").read_text()
    )
    crossover = verification["sample_crossover_db"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    severity = np.linspace(-10, 40, 501)
    variance = sample_power * 10 ** (-severity / 10)
    for metric, label, color, marker in [
        ("no_filter_mse", "No filter: measured mean", "tab:orange", "o"),
        ("filter_mse", "Frozen filter: measured mean", "tab:blue", "s"),
    ]:
        selected = [
            next(
                row
                for row in rows
                if int(row["condition_id"]) == cid and row["metric"] == metric
            )
            for cid in order
        ]
        snr = np.array([float(row["target_snr_db"]) for row in selected])
        mean = np.array([float(row["mean"]) for row in selected])
        low = np.array([float(row["ci95_low"]) for row in selected])
        high = np.array([float(row["ci95_high"]) for row in selected])
        axes[0].errorbar(
            snr,
            mean,
            yerr=np.array([mean - low, high - mean]),
            fmt=marker,
            capsize=4,
            color=color,
            label=label,
        )
    axes[0].plot(
        severity,
        variance * untreated_gain,
        "--",
        color="tab:orange",
        alpha=0.7,
        label="No filter: expected MSE",
    )
    axes[0].plot(
        severity,
        distortion + variance * gain,
        "--",
        color="tab:blue",
        alpha=0.7,
        label="Filter: expected MSE",
    )
    axes[0].axhline(
        distortion, color="gray", ls=":", label="Clean-filter floor"
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("48 kHz full-record MSE\n(NPY amplitude units squared)")
    axes[0].set_title("New phase-1 noise; fixed coefficients and boundaries")
    axes[0].legend(loc="upper right")
    selected = [
        next(
            row
            for row in rows
            if int(row["condition_id"]) == cid
            and row["metric"] == "paired_gain_db"
        )
        for cid in order
    ]
    snr = np.array([float(row["target_snr_db"]) for row in selected])
    mean = np.array([float(row["mean"]) for row in selected])
    low = np.array([float(row["ci95_low"]) for row in selected])
    high = np.array([float(row["ci95_high"]) for row in selected])
    axes[1].errorbar(
        snr,
        mean,
        yerr=np.array([mean - low, high - mean]),
        fmt="o-",
        color="tab:green",
        capsize=4,
        label="Mean paired gain; pointwise 95% intervals",
    )
    axes[1].axhline(0, color="black", lw=1)
    for ax in axes:
        ax.axvspan(0, 30, color="tab:blue", alpha=0.04)
        ax.axvline(crossover, color="gray", ls=":", lw=1.5)
        ax.set_xlim(-12, 42)
    axes[1].text(
        33.6,
        7.0,
        "Sample-domain prediction\n34.79 dB",
        ha="right",
        fontsize=11,
    )
    axes[1].annotate(
        "ID baseline: 2.661 dB",
        xy=(snr[2], mean[2]),
        xytext=(6, 8.5),
        arrowprops={"arrowstyle": "->", "color": "gray"},
        fontsize=11,
    )
    axes[1].set_ylabel("Paired error-SNR gain (dB)")
    axes[1].set_xlabel("Target sample SNR (dB); higher means less added noise")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(figures / f"r23_recovery_quality.{suffix}", dpi=220)
    plt.close(fig)

    personal = np.genfromtxt(
        ROOT / "results/example_c5_r0_48000hz.csv", delimiter=",", names=True
    )
    high = np.genfromtxt(
        ROOT / "results/example_c7_r0_48000hz.csv", delimiter=",", names=True
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    interior = (personal["time_s"] >= 0.5) & (personal["time_s"] < 0.51)
    time_ms = 1000 * (personal["time_s"][interior] - 0.5)
    for key, label, color in [
        ("no_filter", "No filter", "tab:orange"),
        ("filtered", "Frozen filter", "tab:blue"),
        ("reference", "Reference", "black"),
    ]:
        axes[0, 0].plot(
            time_ms, personal[key][interior], color=color, label=label, lw=1
        )
    axes[0, 0].set_title("ID baseline, repeat 0: recovered waveform")
    axes[0, 0].set_ylabel("NPY amplitude units")
    for key, label, color in [
        ("no_filter_error", "No-filter error", "tab:orange"),
        ("filter_error", "Filtered error", "tab:blue"),
    ]:
        axes[0, 1].plot(
            time_ms, personal[key][interior], color=color, label=label, lw=1
        )
    axes[0, 1].set_title("Same record and interval: recovery error")
    for ax in axes[0]:
        ax.set_xlabel("Time from 0.5 s (ms)")
        ax.legend(loc="upper right", fontsize=10)
    for ax, indices, origin, label in [
        (axes[1, 0], slice(0, 768), 0, "First 16 ms"),
        (axes[1, 1], slice(-768, None), 1.984, "Last 16 ms"),
    ]:
        for key, name, color in [
            ("no_filter_error", "No-filter error", "tab:orange"),
            ("filter_error", "Filtered error", "tab:blue"),
        ]:
            ax.plot(
                1000 * (high["time_s"][indices] - origin),
                high[key][indices],
                label=name,
                color=color,
                lw=1,
            )
        ax.set_title(f"40 dB, repeat 0: {label}")
        ax.set_xlabel("Time within endpoint window (ms)")
        ax.set_ylim(-0.65, 0.65)
        ax.legend(
            loc="upper left" if origin > 0 else "upper right", fontsize=10
        )
    axes[1, 0].set_ylabel("Recovery error (NPY units)")
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(figures / f"r23_examples_boundaries.{suffix}", dpi=220)
    plt.close(fig)
    print("R2.3: two figures drawn from saved CSV.")


if __name__ == "__main__":
    main()
