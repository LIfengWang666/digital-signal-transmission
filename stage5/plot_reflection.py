"""Visualise verified R5 decision gains and the alternative rate constraint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from stage4.r43.assessment import read_rows

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def save(fig: plt.Figure, name: str) -> None:
    """Save reproducible raster and vector versions of a measured figure.

    Args:
        fig: Complete figure with labelled units and coverage.
        name: Stable output basename.

    Returns:
        None. Save PNG/SVG without SVG-ID randomness or timestamps.
    """
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(
        folder / f"{name}.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "Matplotlib DST R5"},
    )
    fig.savefig(
        folder / f"{name}.svg", bbox_inches="tight", metadata={"Date": None}
    )
    plt.close(fig)


def plot_limiting_decision() -> None:
    """Show main/personal paired gains with original feasibility explicit.

    Returns:
        None. Negative and small gains remain visible on symmetric-log axes.
    """
    rows = [
        r
        for r in read_rows(HERE / "results/decision_evidence.csv")
        if r["id"] != "base"
    ]
    labels = {
        "band60": "BW 60 Hz*",
        "band80": "BW 80 Hz*",
        "taps257": "257 taps*",
        "taps513": "513 taps",
        "bits16": "16 bit",
        "h1511": "Hamming(15,11)",
        "repeat5": "Repeat(5,1)*",
        "uncoded": "Uncoded",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8, 5.2), sharey=True)
    y = np.arange(len(rows))
    for ax, prefix, title in zip(
        axes,
        ["main", "personal"],
        ["Main objective gain", "Personal-point gain"],
    ):
        for i, row in enumerate(rows):
            mean = float(row[f"{prefix}_gain"])
            lower, upper = float(row[f"{prefix}_gain_lower"]), float(
                row[f"{prefix}_gain_upper"]
            )
            feasible = row["original_constraints_feasible"] == "True"
            ax.errorbar(
                mean,
                i,
                xerr=[[mean - lower], [upper - mean]],
                fmt="o" if feasible else "x",
                color="#0072B2" if feasible else "#777777",
                markersize=6,
                capsize=3,
                linewidth=1.3,
            )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xscale("symlog", linthresh=0.001)
        lower_limits = [float(row[f"{prefix}_gain_lower"]) for row in rows]
        upper_limits = [float(row[f"{prefix}_gain_upper"]) for row in rows]
        ax.set_xlim(min(lower_limits) * 1.8, max(upper_limits) * 1.8)
        ticks = (
            [-1, -0.01, 0, 0.01, 1]
            if prefix == "main"
            else [-0.1, -0.001, 0, 0.001]
        )
        ax.set_xticks(ticks)
        ax.set_xlabel("Baseline NMSE - alternative NMSE")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, [labels[row["id"]] for row in rows])
    axes[0].invert_yaxis()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(
        0.5,
        0.01,
        "Study 5 confirmation | approximate 19-family paired mean intervals\n"
        "*Fails gain or redundancy condition; retained as a diagnostic",
        ha="center",
        fontsize=8.5,
    )
    save(fig, "r51_counterfactual_gains")


def plot_alternative_constraint() -> None:
    """Compare measured quality and exact sent rate under the new320kbit/s cap.

    Returns:
        None. The quality threshold applies only to the declared personal
        point; unavailable or budget-ineligible candidates are not invented.
    """
    rows = read_rows(HERE / "results/constraint_tradeoff.csv")
    assessed = [
        next(r for r in rows if r["variant"] == name)
        for name in ["base", "h1511", "uncoded"]
    ]
    rule = json.loads((HERE / "design.json").read_text())[
        "R52_prespecified_constraint"
    ]
    threshold = rule["primary_NMSE_threshold"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.fill_between(
        [200, 320],
        0,
        threshold,
        color="#009E73",
        alpha=0.10,
        label="Within rate and mean-NMSE limits",
    )
    for row, label, offset in zip(
        assessed,
        ["Hamming(7,4)", "Hamming(15,11)", "Uncoded"],
        [(-100, 18), (-65, -17), (10, -8)],
    ):
        x, mean = float(row["logical_rate_bits_s"]) / 1000, float(
            row["mean_nmse"]
        )
        half = float(row["mean_half_width"])
        good = (
            row["rate_feasible"] == "1" and row["mean_quality_passes"] == "1"
        )
        ax.errorbar(
            x,
            mean,
            yerr=half,
            fmt="o",
            color="#009E73" if good else "#D55E00",
            capsize=4,
            markersize=7,
        )
        ax.annotate(
            label,
            (x, mean),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
        )
    ax.axvline(
        320, color="0.3", linestyle="--", linewidth=1.1, label="320 kbit/s cap"
    )
    ax.axhline(
        threshold,
        color="0.3",
        linestyle=":",
        linewidth=1.1,
        label="Unchanged personal-point NMSE tolerance",
    )
    ax.set_xlim(200, 420)
    ax.set_ylim(0, max(float(r["mean_upper"]) for r in assessed) * 1.18)
    ax.set_xlabel("Logical sent rate (kbit/s; two-second payload)")
    ax.set_ylabel("Mean full-record NMSE")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    save(fig, "r52_quality_rate")


def main() -> None:
    """Draw the requested reflection part only after independent confirmation.

    Returns:
        None. R5.2 is plotted only after its separate constraint assessment.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["r51", "r52", "all"], default="all")
    args = parser.parse_args()
    check = json.loads(
        (HERE / "results/phase1_independent_checks.json").read_text()
    )
    assert check["status"] == "passed"
    plt.rcParams.update(
        {
            "font.size": 11,
            "svg.hashsalt": "DST-R5",
            "savefig.facecolor": "white",
        }
    )
    if args.part in {"r51", "all"}:
        plot_limiting_decision()
    if args.part in {"r52", "all"}:
        plot_alternative_constraint()
    print(f"R5 measured {args.part} figures generated.")


if __name__ == "__main__":
    main()
