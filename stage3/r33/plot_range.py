"""Draw the R3.3 range evidence from saved data and exact reference models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stage3.r33.error_range import (  # noqa: E402
    binary_entropy,
    expectations,
    read_rows,
)

ROOT = Path(__file__).resolve().parent


def save_figure(figure: plt.Figure, name: str) -> None:
    """Save deterministic raster and vector outputs.

    Args:
        figure: Completed scientific figure.
        name: Filename stem.

    Returns:
        None. Write two formats and close the figure.
    """
    figure.savefig(ROOT / "figures" / (name + ".png"), dpi=300)
    figure.savefig(ROOT / "figures" / (name + ".svg"), metadata={"Date": None})
    plt.close(figure)


def main() -> None:
    """Plot exposure precision and the two distinct theoretical boundaries.

    Returns:
        None. Write two figure groups.
    """
    (ROOT / "figures").mkdir(exist_ok=True)
    rows = read_rows(ROOT / "results/summary.csv")
    frozen = json.loads((ROOT / "results/range_freeze.json").read_text())
    personal = frozen["personal_probability"]
    plt.rcdefaults()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["svg.hashsalt"] = "DST-R33"
    dense = np.geomspace(1e-4, 0.5, 401)
    coded = [
        r
        for r in rows
        if r["code"] == "hamming74" and float(r["probability"]) > 0
    ]
    p = np.array([float(r["probability"]) for r in coded])
    observed = np.array([float(r["block_failure_fraction"]) for r in coded])
    low = np.array([float(r["block_ci95_low"]) for r in coded])
    high = np.array([float(r["block_ci95_high"]) for r in coded])
    figure, axes = plt.subplots(2, 1, figsize=(9, 7.2))
    axes[0].loglog(
        dense,
        [expectations(v, "hamming74")[1] for v in dense],
        color="black",
        label="Exact Binomial tail P(J > 1)",
    )
    axes[0].errorbar(
        p,
        observed,
        yerr=[observed - low, high - observed],
        fmt="o",
        markersize=4,
        capsize=3,
        label="Measured; pointwise 95% CI",
    )
    axes[0].set(
        ylabel="Hamming block failure probability",
        title="4,480,000 independent codeword exposures at each positive p",
    )
    axes[0].legend(fontsize=8)
    for name, label, colour, marker in [
        ("uncoded", "Uncoded layout", "tab:grey", "o"),
        ("hamming74", "Hamming layout", "tab:blue", "s"),
    ]:
        records = [
            r
            for r in rows
            if r["code"] == name and float(r["probability"]) > 0
        ]
        x = np.array([float(r["probability"]) for r in records])
        value = np.array([float(r["channel_ber"]) for r in records])
        lower = np.array([float(r["channel_ci95_low"]) for r in records])
        upper = np.array([float(r["channel_ci95_high"]) for r in records])
        axes[1].errorbar(
            x,
            value / x,
            yerr=[(value - lower) / x, (upper - value) / x],
            fmt=marker,
            color=colour,
            markersize=4,
            capsize=3,
            label=label,
        )
    axes[1].axhline(1, color="black", linestyle="--", linewidth=0.8)
    axes[1].set(
        xscale="log",
        xlabel="Target probability p",
        ylabel="Measured channel BER / p",
        title="Independent transmitted-bit counts; exact binomial intervals",
    )
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.axvspan(1e-4, 0.1, color="tab:green", alpha=0.08)
        axis.axvline(personal, color="tab:purple", linestyle=":", linewidth=1)
    figure.tight_layout()
    save_figure(figure, "r33_range_precision")

    figure, axes = plt.subplots(2, 1, figsize=(9, 7.2))
    linear = np.linspace(0, 0.5, 501)
    axes[0].plot(
        linear,
        [1 - binary_entropy(v) for v in linear],
        color="black",
        label="BSC capacity C = 1 - H2(p)",
    )
    axes[0].axhline(
        4 / 7,
        color="tab:blue",
        linestyle="--",
        label="Frozen code rate R = 4/7",
    )
    capacity_root = frozen["capacity_equals_code_rate_at"]
    axes[0].axvline(capacity_root, color="tab:red", linestyle=":")
    axes[0].annotate(
        f"C = R at p = {capacity_root:.5f}",
        (capacity_root, 4 / 7),
        xytext=(0.17, 0.78),
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
    )
    axes[0].set(
        ylabel="Information bits / channel use",
        title="Capacity and the frozen short-code rate",
        xlim=(0, 0.5),
        ylim=(0, 1.04),
    )
    axes[0].legend(fontsize=8, loc="upper right")
    ratio = [expectations(v, "hamming74")[0] / v for v in linear[1:]]
    axes[1].plot(
        linear[1:], ratio, color="black", label="Exact decoded BER / p"
    )
    axes[1].scatter(
        p,
        [float(r["data_ber"]) / float(r["probability"]) for r in coded],
        s=20,
        color="tab:blue",
        label="Measured Hamming data BER / p",
    )
    axes[1].axhline(
        1, color="tab:grey", linestyle="--", label="Uncoded expectation"
    )
    ber_root = frozen["exact_data_ber_equals_uncoded_at"]
    axes[1].axvline(ber_root, color="tab:red", linestyle=":")
    axes[1].annotate(
        f"BER crossover at p = {ber_root:.5f}",
        (ber_root, 1),
        xytext=(0.24, 0.45),
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
    )
    axes[1].set(
        xlabel="BSC probability p",
        ylabel="Data BER / p",
        title="Stress points bracket loss of BER advantage",
        xlim=(0, 0.5),
    )
    axes[1].legend(fontsize=8, loc="lower right")
    for axis in axes:
        axis.axvspan(1e-4, 0.1, color="tab:green", alpha=0.08)
    figure.tight_layout()
    save_figure(figure, "r33_capacity_ber_limits")
    print("R3.3: range/precision and capacity/BER-boundary figures prepared.")


if __name__ == "__main__":
    main()
