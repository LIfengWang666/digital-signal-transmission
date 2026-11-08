"""Plot the recorded R3.2 pilot profile, reconstruction and cost evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
NAMES = {
    "uncoded": "Uncoded",
    "hamming74": "Hamming (7,4)",
    "hamming1511": "Hamming (15,11)",
    "repetition51": "Repetition (5,1)",
}
COLOURS = ["tab:grey", "tab:blue", "tab:orange", "tab:green"]


def read_rows(name: str) -> list[dict[str, str]]:
    """Read saved evidence.

    Args:
        name: CSV filename in results.

    Returns:
        Header-keyed string rows.
    """
    with (ROOT / "results" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(figure: plt.Figure, name: str) -> None:
    """Save deterministic raster and vector figures.

    Args:
        figure: Completed scientific figure.
        name: Output filename stem.

    Returns:
        None. Save two formats and close the figure.
    """
    figure.savefig(ROOT / "figures" / (name + ".png"), dpi=300)
    figure.savefig(ROOT / "figures" / (name + ".svg"), metadata={"Date": None})
    plt.close(figure)


def main() -> None:
    """Draw three figure groups without changing any measurement.

    Returns:
        None. Write PNG/SVG artifacts.
    """
    (ROOT / "figures").mkdir(exist_ok=True)
    summary = read_rows("summary.csv")
    profile = read_rows("error_profile.csv")
    selection = read_rows("code_selection.csv")
    protocol = json.loads((ROOT / "design.json").read_text())
    plt.rcdefaults()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["svg.hashsalt"] = "DST-R32"
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for axis, code, colour in zip(axes, list(NAMES)[1:], COLOURS[1:]):
        rows = [
            r
            for r in profile
            if r["code"] == code and float(r["probability"]) == 0.01
        ]
        observed, theory = [], []
        for weight in range(4):
            group = [r for r in rows if min(int(r["weight"]), 3) == weight]
            observed.append(sum(float(r["observed_fraction"]) for r in group))
            theory.append(sum(float(r["binomial_probability"]) for r in group))
        axis.bar(
            np.arange(4) - 0.18, observed, 0.36, color=colour, label="Measured"
        )
        axis.bar(
            np.arange(4) + 0.18,
            theory,
            0.36,
            color="none",
            edgecolor="black",
            label="Binomial",
        )
        axis.set(
            yscale="log",
            xticks=range(4),
            xticklabels=["0", "1", "2", "3+"],
            xlabel="Flipped bits in one transmitted block",
            title=NAMES[code],
        )
    axes[0].set_ylabel("Fraction of blocks (log scale)")
    axes[0].legend(fontsize=8)
    figure.suptitle("Actual iid BSC error profile at p = 0.01; 40 masks/code")
    figure.tight_layout()
    save_figure(figure, "r32_block_profile")

    figure, axes = plt.subplots(2, 1, figsize=(9, 7))
    for (code, label), colour in zip(NAMES.items(), COLOURS):
        rows = [
            r
            for r in summary
            if r["code"] == code and float(r["probability"]) > 0
        ]
        p = np.array([float(r["probability"]) for r in rows])
        theory = np.array([float(r["theory_data_ber"]) for r in rows])
        observed = np.array([float(r["data_ber"]) for r in rows])
        axes[0].loglog(p, theory, "-", color=colour, label=label)
        positive = observed > 0
        axes[0].loglog(
            p[positive], observed[positive], "o", color=colour, markersize=4
        )
        axes[1].loglog(
            p,
            [float(r["mean_mse_48000"]) for r in rows],
            "o-",
            color=colour,
            label=label,
            markersize=4,
        )
    axes[0].set(
        ylabel="Decoded data BER",
        title="Lines: exact pattern enumeration; dots: measured nonzero BER",
    )
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].text(
        0.02,
        0.03,
        "Repetition: 0/17,920,000 wrong bits at p = 0.0001 and 0.001\n"
        "Zero observations omitted on log axis; theory remains positive.",
        transform=axes[0].transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    axes[1].axhline(
        float(summary[0]["mean_prechannel_mse_48000"]),
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Before BSC",
    )
    axes[1].set(
        xlabel="BSC probability p",
        ylabel="Full-record MSE (NPY units squared)",
        title="48 kHz grid; five Gaussian banks, eight masks per bank",
    )
    axes[1].legend(fontsize=8, ncol=2)
    figure.tight_layout()
    save_figure(figure, "r32_ber_reconstruction")

    figure, axes = plt.subplots(2, 1, figsize=(9, 7))
    for row, colour in zip(selection, COLOURS):
        code = row["code"]
        mean = float(row["score"])
        low = float(row["score_ci95_low"])
        high = float(row["score_ci95_high"])
        expansion = float(row["payload_expansion"])
        axes[0].errorbar(
            expansion,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            color=colour,
            capsize=4,
            label=NAMES[code],
        )
        axes[0].annotate(
            f"{float(row['transmitted_bits_per_second'])/1000:g} kbit/s",
            (expansion, mean),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].axvline(
        protocol["max_payload_expansion"],
        color="black",
        linestyle="--",
        label="Project expansion budget: 2",
    )
    axes[0].set(
        xlabel="Transmitted bits / source bits",
        ylabel="Normalised MSE score J",
        title="Lower score is better; intervals across five input banks",
        xlim=(0.8, 5.8),
        yscale="log",
    )
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    waveform = np.genfromtxt(
        ROOT / "results/example_48000hz.csv", delimiter=",", names=True
    )
    start, stop = protocol["example"]["plot_window_s"]
    window = (waveform["time_s"] >= start) & (waveform["time_s"] < stop)
    for field, label, colour, style in [
        ("reference", "Original reference", "black", "-"),
        ("prechannel", "Before BSC", "tab:green", "--"),
        ("uncoded", "Uncoded", "tab:grey", "-"),
        ("hamming74", "Hamming (7,4)", "tab:blue", "-"),
    ]:
        axes[1].plot(
            waveform["time_s"][window] * 1000,
            waveform[field][window],
            style,
            color=colour,
            label=label,
            linewidth=0.8,
        )
    axes[1].set(
        xlabel="Time (ms)",
        ylabel="NPY amplitude units",
        title="Predeclared example: p = 0.003, Gaussian bank 0, BSC repeat 0",
    )
    axes[1].legend(fontsize=8, ncol=2)
    figure.tight_layout()
    save_figure(figure, "r32_cost_example")
    print("R3.2: three scientific figure groups written from saved CSV.")


if __name__ == "__main__":
    main()
