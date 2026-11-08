"""Plot the frozen-rule independent R3.4 coding comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from stage3.r33.error_range import expectations, read_rows  # noqa: E402

ROOT = Path(__file__).resolve().parent


def save_figure(
    figure: plt.Figure, name: str, horizontal: bool = False
) -> None:
    """Save deterministic scientific raster and vector versions.

    Args:
        figure: Finished figure.
        name: Output stem.
        horizontal: Export an additional two-column report layout.

    Returns:
        None. Save and close.
    """
    if horizontal:
        name += "_horizontal"
        for axis in figure.axes:
            axis.set_title(textwrap.fill(axis.get_title(), 34), fontsize=10)
            axis.set_xlabel("BSC probability p", fontsize=11)
            axis.yaxis.label.set_fontsize(11)
            axis.tick_params(labelsize=10)
            legend = axis.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(9)
        if "bit_recovery" in name:
            figure.axes[1].set_ylabel("Bit-perfect payload fraction")
        figure.tight_layout(pad=1.0, w_pad=1.5)
    png = ROOT / "figures" / (name + ".png")
    figure.savefig(png, dpi=300)
    figure.savefig(ROOT / "figures" / (name + ".svg"), metadata={"Date": None})
    if horizontal:
        report_images = ROOT.parents[1] / "report/images"
        report_images.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, report_images / png.name)
    plt.close(figure)


def main(horizontal: bool = False) -> None:
    """Draw quality, data recovery and sensitivity evidence from saved CSV.

    Args:
        horizontal: Export the two additional side-by-side report figures.

    Returns:
        None. Write three figure groups without changing the experiment.
    """
    folder = ROOT / "results"
    (ROOT / "figures").mkdir(exist_ok=True)
    design = json.loads((folder / "validation_freeze.json").read_text())
    protocol = json.loads((ROOT / "design.json").read_text())
    rows = read_rows(folder / "summary.csv")
    sensitivity = read_rows(folder / "threshold_sensitivity.csv")
    plt.rcdefaults()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["svg.hashsalt"] = "DST-R34"
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11})
    tau = design["primary_nmse_threshold"]
    figure, axes = plt.subplots(
        1 if horizontal else 2,
        2 if horizontal else 1,
        figsize=(8.4, 3.6) if horizontal else (9, 7.4),
    )
    for code, label, colour in [
        ("uncoded", "Uncoded", "tab:grey"),
        ("hamming74", "Hamming (7,4)", "tab:blue"),
    ]:
        data = [
            r
            for r in rows
            if r["code"] == code and float(r["probability"]) > 0
        ]
        p = np.array([float(r["probability"]) for r in data])
        mean = np.array([float(r["mean_nmse"]) for r in data])
        low = np.array([float(r["adjusted_ci_low"]) for r in data])
        high = np.array([float(r["adjusted_ci_high"]) for r in data])
        axes[0].errorbar(
            p,
            mean,
            yerr=[mean - low, high - mean],
            fmt="o-",
            color=colour,
            capsize=3,
            markersize=4,
            label=label,
        )
    axes[0].set(
        xscale="log",
        yscale="log",
        ylabel="Full-record NMSE",
        title="300 new input records; adjusted intervals over 30 means",
    )
    axes[0].axhline(
        tau,
        color="tab:red",
        linestyle="--",
        label="Fixed 2x design-baseline budget",
    )
    axes[0].legend(fontsize=9)
    data = [
        r
        for r in rows
        if r["code"] == "hamming74" and 0 < float(r["probability"]) <= 0.005
    ]
    colours = {
        "pass": "tab:green",
        "fail": "tab:red",
        "inconclusive": "tab:orange",
    }
    for row in data:
        p, mean = float(row["probability"]), float(row["mean_nmse"])
        low, high = float(row["adjusted_ci_low"]), float(
            row["adjusted_ci_high"]
        )
        axes[1].errorbar(
            p,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            capsize=4,
            color=colours[row["status"]],
            markersize=5,
        )
    zero = next(
        r
        for r in rows
        if r["code"] == "hamming74" and float(r["probability"]) == 0
    )
    axes[1].axhline(
        tau, color="tab:red", linestyle="--", label="Fixed threshold"
    )
    axes[1].axhline(
        float(zero["mean_nmse"]),
        color="black",
        linestyle=":",
        label="New p=0 validation mean",
    )
    axes[1].set(
        xscale="log",
        xlabel="BSC probability p",
        ylabel="Hamming NMSE",
        title="Transition detail: green = pass; red = fail",
        ylim=(0.02, 0.135),
    )
    axes[1].legend(fontsize=9)
    figure.tight_layout()
    save_figure(figure, "r34_quality_gain", horizontal)

    dense = np.geomspace(1e-4, 0.5, 401)
    figure, axes = plt.subplots(
        1 if horizontal else 2,
        2 if horizontal else 1,
        figsize=(8.4, 3.6) if horizontal else (9, 7.4),
    )
    for code, label, colour in [
        ("uncoded", "Uncoded", "tab:grey"),
        ("hamming74", "Hamming (7,4)", "tab:blue"),
    ]:
        data = [
            r
            for r in rows
            if r["code"] == code and float(r["probability"]) > 0
        ]
        positive = [r for r in data if float(r["data_ber"]) > 0]
        axes[0].loglog(
            dense,
            [expectations(p, code)[0] for p in dense],
            color=colour,
            label=label + ": exact expectation",
        )
        axes[0].loglog(
            [float(r["probability"]) for r in positive],
            [float(r["data_ber"]) for r in positive],
            "o",
            color=colour,
            markersize=4,
        )
        bit_rows = [r for r in data if float(r["probability"]) <= 0.005]
        p = np.array([float(r["probability"]) for r in bit_rows])
        mean = np.array(
            [float(r["perfect_payload_fraction"]) for r in bit_rows]
        )
        low = np.array(
            [float(r["perfect_payload_ci95_low"]) for r in bit_rows]
        )
        high = np.array(
            [float(r["perfect_payload_ci95_high"]) for r in bit_rows]
        )
        axes[1].errorbar(
            p,
            mean,
            yerr=[mean - low, high - mean],
            fmt="o",
            color=colour,
            capsize=3,
            markersize=4,
            label=label + ": measured",
        )
        exact_p = np.geomspace(1e-4, 0.005, 301)
        blocks = 448000 if code == "uncoded" else 112000
        exact = [
            math.exp(blocks * math.log1p(-expectations(p, code)[1]))
            for p in exact_p
        ]
        axes[1].plot(exact_p, exact, color=colour)
    axes[0].set(
        ylabel="Decoded real-data BER",
        title="Lines: exact decoder expectations; dots: validation counts",
    )
    axes[0].legend(fontsize=9)
    axes[1].set(
        xscale="log",
        xlabel="BSC probability p",
        ylabel="Bit-perfect complete-payload fraction",
        title="Whole-payload recovery; pointwise exact 95% intervals",
        ylim=(-0.04, 1.04),
    )
    axes[1].legend(fontsize=9)
    figure.tight_layout()
    save_figure(figure, "r34_bit_recovery", horizontal)
    if horizontal:
        assets = [
            ROOT / "figures" / (stem + "_horizontal." + suffix)
            for stem in ["r34_quality_gain", "r34_bit_recovery"]
            for suffix in ["png", "svg"]
        ]
        record = {
            "status": "passed",
            "purpose": "Presentation only; same CSV, limits and intervals",
            "source_sha256": {
                p.relative_to(ROOT.parents[1])
                .as_posix(): hashlib.sha256(p.read_bytes())
                .hexdigest()
                for p in [
                    folder / "summary.csv",
                    folder / "validation_freeze.json",
                ]
            },
            "asset_sha256": {
                p.relative_to(ROOT.parents[1])
                .as_posix(): hashlib.sha256(p.read_bytes())
                .hexdigest()
                for p in assets
            },
            "report_image_copies_sha256": {
                p.relative_to(ROOT.parents[1])
                .as_posix(): hashlib.sha256(p.read_bytes())
                .hexdigest()
                for p in [
                    ROOT.parents[1] / "report/images" / p.name
                    for p in assets
                    if p.suffix == ".png"
                ]
            },
            "plotting_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "new_measurements_or_independent_records": 0,
        }
        destination = ROOT.parents[1] / "stage6/results/editorial_figures.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print("R3.4: two horizontal layouts generated from unchanged CSV.")
        return

    points = sorted(design["conditions"], key=lambda r: r["probability"])
    matrix = []
    for factor in design["threshold_sensitivity_factors"]:
        codes = []
        for point in points:
            row = next(
                r
                for r in sensitivity
                if float(r["budget_factor"]) == factor
                and int(r["condition_id"]) == point["condition_id"]
            )
            if "inconclusive" in [row["uncoded_status"], row["coded_status"]]:
                value = 2
            elif row["coding_gain_established"] == "1":
                value = 1
            elif row["uncoded_status"] == row["coded_status"] == "pass":
                value = 3
            else:
                value = 0
            codes.append(value)
        matrix.append(codes)
    palette = ["#d9d9d9", "#80c99b", "#f2c16b", "#91bce0"]
    figure, axes = plt.subplots(
        3, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1.1, 1, 1]}
    )
    axes[0].imshow(
        matrix, aspect="auto", cmap=ListedColormap(palette), vmin=0, vmax=3
    )
    axes[0].set(
        xticks=range(len(points)),
        xticklabels=[
            "ID" if p["condition_id"] == 11 else f"{p['probability']:g}"
            for p in points
        ],
        yticks=range(3),
        yticklabels=["1.5x", "2x primary", "4x"],
        ylabel="Total MSE budget",
        title="Predeclared threshold sensitivity",
    )
    axes[0].tick_params(axis="x", rotation=40, labelsize=9)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axes[0].text(
                j,
                i,
                ["F", "G", "?", "B"][value],
                ha="center",
                va="center",
                fontsize=9,
            )
    axes[0].legend(
        handles=[
            Patch(facecolor=c, label=l)
            for c, l in zip(
                palette,
                [
                    "F: no separation",
                    "G: coding gain",
                    "?: inconclusive",
                    "B: both pass",
                ],
            )
        ],
        ncol=4,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.44),
    )
    for axis, example in zip(axes[1:], protocol["examples"]):
        data = np.genfromtxt(
            folder / f"example_{example['name']}_48000hz.csv",
            delimiter=",",
            names=True,
        )
        start, stop = protocol["example_window_s"]
        window = (data["time_s"] >= start) & (data["time_s"] < stop)
        for field, label, colour, style in [
            ("reference", "Original", "black", "-"),
            ("prechannel", "Before BSC", "tab:green", "--"),
            ("uncoded", "Uncoded", "tab:grey", "-"),
            ("hamming74", "Hamming", "tab:blue", "-"),
        ]:
            axis.plot(
                data["time_s"][window] * 1000,
                data[field][window],
                style,
                color=colour,
                linewidth=0.8,
                label=label,
            )
        p = next(
            c["probability"]
            for c in points
            if c["condition_id"] == example["condition_id"]
        )
        axis.set(
            xlabel="Time (ms)",
            ylabel="NPY amplitude units",
            title=f"Predeclared new record 0: p = {p:.7g}",
        )
        axis.legend(fontsize=8, ncol=4, loc="upper right")
    figure.tight_layout()
    save_figure(figure, "r34_sensitivity_examples")
    print("R3.4: three measured figure groups prepared.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizontal", action="store_true")
    main(parser.parse_args().horizontal)
