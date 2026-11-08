"""Plot the declared metric counterexample from saved deterministic CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Draw loss-ranking evidence, explicitly labelled as semantic examples.

    Returns:
        None. Save PNG/SVG figures from the two declared CSV rows.
    """
    with (HERE / "results/semantic_checks.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    names = ["diffuse_0.1", "sparse_2_at_1percent"]
    labels = [
        "Diffuse: +/-0.1\nat every sample",
        "Sparse: +/-2\nat 1% of samples",
    ]
    colors = ["#24759a", "#c26433"]
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for ax, key, ylabel, title in zip(
        axes,
        ["nmse", "mae"],
        ["NMSE (dimensionless)", "MAE (NPY amplitude units)"],
        [
            "Squared loss penalises sparse large errors",
            "Absolute loss prefers the sparse example",
        ],
    ):
        values = [float(rows[name][key]) for name in names]
        ax.bar(np.arange(2), values, color=colors, width=0.55)
        ax.set_xticks(np.arange(2), labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.3 * max(values))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.text(
                i, value + 0.025 * max(values), f"{value:.5f}", ha="center"
            )
    figure.suptitle(
        "Deterministic metric checks: not channel measurements", fontsize=11
    )
    figure.tight_layout()
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ["png", "svg"]:
        figure.savefig(folder / f"r41_metric_ranking.{extension}", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
