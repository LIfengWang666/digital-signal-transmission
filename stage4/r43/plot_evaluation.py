"""Plot real held-out CSV with four curves and explicit uncertainty scope."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

from stage4.r43.assessment import read_rows

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIGS = ["00", "10", "01", "11"]
NAMES = ["Neither (00)", "Filter only (10)", "Code only (01)", "Both (11)"]
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
MARKERS = ["o", "s", "^", "D"]
STYLES = ["-", "--", "-.", ":"]


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a sharp raster figure and deterministic editable vector source.

    Args:
        fig: Fully labelled figure generated from saved CSV.
        name: Stable output basename.

    Returns:
        None. Save PNG and SVG without random SVG IDs or timestamps.
    """
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(
        folder / f"{name}.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "Matplotlib DST R4.3"},
    )
    fig.savefig(
        folder / f"{name}.svg", bbox_inches="tight", metadata={"Date": None}
    )
    plt.close(fig)


def slice_rows(rows: list[dict], choices: dict, name: str) -> list[dict]:
    """Select and sort one of the two frozen degradation slices.

    Args:
        rows: Condition/configuration or effect-summary records.
        choices: Saved conditions with explicit slice membership.
        name: gaussian_at_personal_p or bsc_at_personal_snr.

    Returns:
        Requested rows; the BSC plotting slice includes positive p only.
        Its exact p0 control is displayed separately rather than moved.
    """
    conditions = {c["joint_condition_id"]: c for c in choices["conditions"]}
    selected = []
    for row in rows:
        cid = int(row["joint_condition_id"])
        if (
            cid not in conditions
            or name not in conditions[cid]["slice_membership"]
        ):
            continue
        c = conditions[cid]
        if name == "bsc_at_personal_snr" and c["probability"] == 0:
            continue
        selected.append(
            {
                **row,
                "x": (
                    c["snr_db"]
                    if name.startswith("gaussian")
                    else c["probability"]
                ),
                "primary": c["primary"],
            }
        )
    return sorted(selected, key=lambda row: row["x"])


def central_figure(summary: list[dict], choices: dict) -> None:
    """Show both slices, four configurations and92-family mean intervals.

    Args:
        summary:88 held-out configuration/condition means and limits.
        choices: Frozen physical conditions.

    Returns:
        None. Draw p0 as a separate labelled inset with its true value.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.8))
    personal = next(
        c for c in choices["conditions"] if c["joint_condition_id"] == 11
    )
    for panel, (ax, name) in enumerate(
        zip(axes, ["gaussian_at_personal_p", "bsc_at_personal_snr"])
    ):
        selected = slice_rows(summary, choices, name)
        for config, label, colour, marker, style in zip(
            CONFIGS, NAMES, COLOURS, MARKERS, STYLES
        ):
            rows = [r for r in selected if r["configuration"] == config]
            x = np.array([r["x"] for r in rows])
            mean = np.array([float(r["mean_nmse"]) for r in rows])
            half = np.array([float(r["half_width_nmse"]) for r in rows])
            ax.errorbar(
                x,
                mean,
                yerr=half,
                color=colour,
                linestyle=style,
                marker=marker,
                markersize=5,
                capsize=2.5,
                linewidth=1.6,
                label=label,
            )
            diagnostic = [i for i, r in enumerate(rows) if not r["primary"]]
            ax.scatter(
                x[diagnostic],
                mean[diagnostic],
                facecolors="white",
                edgecolors=colour,
                marker=marker,
                s=34,
                zorder=4,
            )
        # A nonpositive limit is never silently clipped for a logarithmic plot.
        lower = min(float(r["lower_nmse"]) for r in selected)
        if lower > 0:
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.axhline(0, color="0.4", linewidth=0.7)
        ax.set_ylabel("Mean full-record NMSE")
        ax.grid(True, which="major", alpha=0.25)
        if panel == 0:
            ax.set_title(
                f"(a) Gaussian slice: fixed p = {personal['probability']:.6g}",
                loc="left",
            )
            ax.set_xlabel(
                "Target Gaussian SNR (dB; higher = less Gaussian noise)"
            )
            ax.axvline(
                personal["snr_db"], color="0.35", linewidth=0.8, linestyle="--"
            )
            ax.text(
                0.98,
                0.94,
                r"Dashed line: $S_{ID}$",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
            )
        else:
            ax.set_xscale("log")
            ax.set_title(
                f"(b) BSC slice: fixed SNR = {personal['snr_db']:.6f} dB",
                loc="left",
            )
            ax.set_xlabel(
                "Bit-flip probability p (higher = more channel errors)"
            )
            ax.axvline(
                personal["probability"],
                color="0.35",
                linewidth=0.8,
                linestyle="--",
            )
            ax.text(
                0.98,
                0.08,
                r"Dashed line: $p_{ID}$",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
            )
            inset = ax.inset_axes([0.08, 0.62, 0.34, 0.29])
            controls = [
                next(
                    r
                    for r in summary
                    if int(r["joint_condition_id"]) == 0
                    and r["configuration"] == config
                )
                for config in CONFIGS
            ]
            for i, (row, colour, marker) in enumerate(
                zip(controls, COLOURS, MARKERS)
            ):
                inset.errorbar(
                    i,
                    float(row["mean_nmse"]),
                    yerr=float(row["half_width_nmse"]),
                    fmt=marker,
                    color=colour,
                    markersize=4,
                    capsize=2,
                )
            inset.set_yscale("log")
            inset.set_xticks(range(4), ["00", "10", "01", "11"])
            inset.tick_params(labelsize=7)
            inset.set_title("p = 0 control; Gaussian retained", fontsize=7.5)
            inset.grid(alpha=0.2)
            inset.margins(y=0.15)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97), h_pad=2.0)
    fig.text(
        0.5,
        0.005,
        "1400 held-out bundles | approximate 92-family mean intervals\n"
        "Open markers: diagnostic conditions",
        ha="center",
        fontsize=8.2,
    )
    save_figure(fig, "r43_central_quality")


def effect_figure(rows: list[dict], choices: dict) -> None:
    """Show both conditional filter/code effects, including negative gains.

    Args:
        rows: Paired effect means and conservative simultaneous bounds.
        choices: Frozen physical conditions and slice membership.

    Returns:
        None. Preserve zero and negative bounds on a symmetric-log axis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.4))
    for column, name in enumerate(
        ["gaussian_at_personal_p", "bsc_at_personal_snr"]
    ):
        selected = slice_rows(rows, choices, name)
        for row_index, effects in enumerate(
            [
                ("filter_without_code", "filter_with_code"),
                ("code_without_filter", "code_with_filter"),
            ]
        ):
            ax = axes[row_index, column]
            for effect, colour, label, marker in zip(
                effects,
                COLOURS[:2],
                ["Other switch off", "Other switch on"],
                ["o", "s"],
            ):
                chosen = [r for r in selected if r["effect"] == effect]
                mean = np.array([float(r["mean_reduction"]) for r in chosen])
                lower = np.array(
                    [float(r["simultaneous_lower"]) for r in chosen]
                )
                upper = np.array(
                    [float(r["simultaneous_upper"]) for r in chosen]
                )
                ax.errorbar(
                    [r["x"] for r in chosen],
                    mean,
                    yerr=np.array([mean - lower, upper - mean]),
                    color=colour,
                    marker=marker,
                    markersize=4,
                    capsize=2,
                    linewidth=1.2,
                    label=label,
                )
            ax.axhline(0, color="0.25", linewidth=0.8)
            ax.set_yscale("symlog", linthresh=0.001)
            ax.grid(alpha=0.25)
            ax.set_ylabel(
                ("Filter" if row_index == 0 else "Code") + " NMSE reduction"
            )
            if column:
                ax.set_xscale("log")
                ax.set_xlabel("Bit-flip probability p")
            else:
                ax.set_xlabel("Target Gaussian SNR (dB)")
            if row_index == 0:
                ax.set_title("Gaussian slice" if column == 0 else "BSC slice")
            ax.margins(y=0.12)
            ax.legend(
                fontsize=9,
                loc="lower left" if column == 1 else "best",
            )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(
        0.5,
        0.005,
        "Positive = NMSE improvement | bounds from92 primary mean limits\n"
        "Symmetric-log scale preserves negative effects near zero",
        ha="center",
        fontsize=8.3,
    )
    save_figure(fig, "r43_conditional_effects")


def component_figure(rows: list[dict]) -> None:
    """Display powers and their signed cross-term sum at fixed example cases.

    Args:
        rows: Mean five-power/ten-cross-term table, evaluated at48 kHz.

    Returns:
        None. Total MSE is indicated independently of the component stacks.
    """
    fig, grid = plt.subplots(2, 2, figsize=(8, 6.5))
    axes = [grid[0, 0], grid[0, 1], grid[1, 0]]
    colours = ["#777777", "#56B4E9", "#E69F00", "#009E73", "#CC79A7"]
    for ax, cid, title in zip(
        axes,
        [11, 5, 21],
        [
            "Personal point",
            "Personal SNR; p = 0.03",
            "SNR = 40 dB; personal p",
        ],
    ):
        chosen = [
            next(
                r
                for r in rows
                if int(r["joint_condition_id"]) == cid
                and r["configuration"] == config
            )
            for config in CONFIGS
        ]
        bottom = np.zeros(4)
        for name, colour in zip(
            ["sampling", "filter", "gaussian", "quantisation", "channel"],
            colours,
        ):
            values = np.array([float(r[f"power_{name}"]) for r in chosen])
            ax.bar(
                range(4),
                values,
                bottom=bottom,
                label=name.capitalize(),
                color=colour,
                width=0.64,
            )
            bottom += values
        cross = np.array(
            [
                sum(float(v) for k, v in r.items() if k.startswith("cross_"))
                for r in chosen
            ]
        )
        ax.bar(
            range(4),
            cross,
            bottom=bottom,
            width=0.64,
            facecolor="none",
            edgecolor="0.2",
            hatch="///",
            label="Signed sum of 10 cross terms",
        )
        ax.scatter(
            range(4),
            [float(r["mean_mse_from_components"]) for r in chosen],
            color="black",
            s=22,
            label="Total MSE",
            zorder=4,
        )
        ax.set_xticks(range(4), CONFIGS)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Configuration")
        ax.axhline(0, color="0.3", linewidth=0.7)
        ax.grid(axis="y", alpha=0.2)
        upper = max(float(np.max(bottom)), float(np.max(bottom + cross)))
        lower = min(0.0, float(np.min(bottom + cross)))
        ax.set_ylim(lower - (0.03 * upper if lower < 0 else 0), 1.12 * upper)
    axes[0].set_ylabel("Mean error power")
    axes[2].set_ylabel("Mean error power")
    handles, labels = axes[0].get_legend_handles_labels()
    grid[1, 1].axis("off")
    grid[1, 1].legend(
        handles, labels, loc="center", fontsize=9.5, frameon=False
    )
    fig.tight_layout()
    save_figure(fig, "r43_error_components")


def example_figure() -> None:
    """Plot the three prespecified record-zero200--220ms windows.

    Returns:
        None. Windows are fixed before validation and do not replace
        the full two-second fidelity measurements.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    for ax, name, title in zip(
        axes,
        ["personal", "personal_snr_p003", "snr40_personal_p"],
        [
            "Personal point",
            "Personal SNR; p = 0.03",
            "SNR = 40 dB; personal p",
        ],
    ):
        rows = read_rows(HERE / "results" / f"example_{name}.csv")
        time = np.array([float(r["time_s"]) for r in rows]) * 1000
        ax.plot(
            time,
            [float(r["reference"]) for r in rows],
            color="black",
            linewidth=1.8,
            label="Reference",
        )
        for config, label, colour, style in zip(
            CONFIGS, NAMES, COLOURS, STYLES
        ):
            ax.plot(
                time,
                [float(r[f"reconstruction_{config}"]) for r in rows],
                color=colour,
                linestyle=style,
                linewidth=1.0,
                alpha=0.9,
                label=label,
            )
        ax.set_title(title + "; held-out record 0", loc="left", fontsize=10)
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (ms); prespecified local window")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=3,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "r43_fixed_examples")


def main() -> None:
    """Load verified CSV and generate central and supporting R4.3 figures.

    Returns:
        None. Never fabricate, smooth or interpolate untested conditions.
    """
    checked = json.loads(
        (HERE / "results/independent_checks.json").read_text()
    )
    assert (
        checked["status"] == "passed"
        and checked["independent_path_records"] == 123200
    )
    choices = json.loads(
        (HERE / "results/validation_freeze.json").read_text()
    )["choices"]
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 11,
            "svg.hashsalt": "DST-R43",
            "savefig.facecolor": "white",
        }
    )
    central_figure(read_rows(HERE / "results/summary.csv"), choices)
    effect_figure(read_rows(HERE / "results/effect_summary.csv"), choices)
    component_figure(read_rows(HERE / "results/component_summary.csv"))
    example_figure()
    print("R4.3: central four-curve panels and support figures generated.")


if __name__ == "__main__":
    main()
