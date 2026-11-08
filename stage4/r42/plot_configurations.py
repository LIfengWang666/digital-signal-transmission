"""Show what the paired configuration pilot checks at the personal point."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from reproduction.prepare_r21_report import read_rows

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Plot descriptive phase-0 means without claiming final validation.

    Returns:
        None. Save PNG/SVG with shared labels/colours for the four paths.
    """
    rows = read_rows(HERE / "results/pilot_summary.csv")
    point = {
        r["configuration"]: r for r in rows if r["joint_condition_id"] == "11"
    }
    configs = ["00", "10", "01", "11"]
    labels = ["00: Neither", "10: Filter only", "01: Code only", "11: Both"]
    colors = ["#777777", "#24759a", "#c26433", "#3b8c65"]
    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "DST-R42",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    values = [
        [float(point[c]["mean_nmse"]) for c in configs],
        [float(point[c]["data_ber"]) * 448000 for c in configs],
    ]
    for ax, data, ylabel in zip(
        axes,
        values,
        ["Mean full-record NMSE", "Mean wrong bits per 448000-bit payload"],
    ):
        ax.bar(np.arange(4), data, color=colors, width=0.62)
        ax.set_xticks(np.arange(4), labels, rotation=20, ha="right")
        ax.set_ylim(0, max(data) * 1.25)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for i, value in enumerate(data):
            ax.text(i, value + 0.025 * max(data), f"{value:.4g}", ha="center")
    axes[0].set_title("Filtering and ECC alter waveform error")
    axes[1].set_title("Shared-mask filter pairs have the same bit errors")
    figure.suptitle(
        "Personal SNR and p: 20 paired phase-0 bundles (not validation)",
        fontsize=11,
    )
    figure.tight_layout()
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    figure.savefig(folder / "r42_paired_checks.png", dpi=220)
    figure.savefig(folder / "r42_paired_checks.svg", metadata={"Date": None})
    plt.close(figure)


if __name__ == "__main__":
    main()
