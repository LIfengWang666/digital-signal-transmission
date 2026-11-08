"""Draw R3.1 evidence from saved CSV, following the quantisation example plot questions."""

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
    """Read one saved R3.1 CSV without regenerating its measurements.

    Args:
        name: Filename under results.

    Returns:
        Header-keyed records.
    """
    with (ROOT / "results" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(figure: plt.Figure, name: str) -> None:
    """Save reproducible raster and vector versions of one figure.

    Args:
        figure: Finished figure.
        name: Filename stem under figures.

    Returns:
        None. Save and close the figure.
    """
    figure.savefig(ROOT / "figures" / (name + ".png"), dpi=300)
    figure.savefig(ROOT / "figures" / (name + ".svg"), metadata={"Date": None})
    plt.close(figure)


def main() -> None:
    """Produce three figure groups from the actual design evidence.

    Returns:
        None. Write PNG/SVG figures.
    """
    (ROOT / "figures").mkdir(exist_ok=True)
    frozen = json.loads((ROOT / "results/quantiser_freeze.json").read_text())
    summary = read_rows("summary.csv")
    selection = read_rows("bit_selection.csv")
    trials = read_rows("trials.csv")
    bits = frozen["bits"]
    a = frozen["half_range"]
    step = frozen["step"]
    plt.rcdefaults()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["svg.hashsalt"] = "DST-R31"
    figure, axes = plt.subplots(2, 1, figsize=(9, 7.2))
    personal = [
        r
        for r in summary
        if r["condition_id"] == "5" and r["branch"] == "filtered"
    ]
    widths = [int(r["bits"]) for r in personal]
    measured = np.array([float(r["mean_sqnr_db"]) for r in personal])
    axes[0].plot(widths, measured, "o-", label="Measured mean SQNR")
    axes[0].fill_between(
        widths,
        [float(r["min_sqnr_db"]) for r in personal],
        [float(r["max_sqnr_db"]) for r in personal],
        alpha=0.2,
        label="Range across five design records",
    )
    axes[0].plot(
        widths,
        [float(r["mean_model_sqnr_db"]) for r in personal],
        "--",
        color="black",
        label="Model using actual endpoint spacing",
    )
    axes[0].set(
        ylabel="SQNR (dB)",
        title="Personal SNR 2.661 dB; frozen filter; fixed range [-18,18]",
    )
    axes[0].legend(fontsize=9)
    ratios = [100 * float(r["worst_component_ratio"]) for r in selection]
    axes[1].semilogy(widths, ratios, "o-", color="tab:purple")
    axes[1].axhline(
        1, color="black", linestyle="--", label="Declared 1% component budget"
    )
    axes[1].set(
        xlabel="Bits per sample",
        ylabel="Worst quantisation/upstream error power (%)",
        title="Both branches, eight SNRs, five records each; 48 kHz grid",
        xticks=widths,
    )
    for axis in axes:
        axis.axvline(bits, color="tab:red", linestyle=":", label=None)
    axes[1].legend(fontsize=9)
    figure.tight_layout()
    save_figure(figure, "r31_sqnr_selection")

    waveform = np.loadtxt(
        ROOT / "results/example_waveform.csv", delimiter=",", skiprows=1
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    window = slice(0, 80)
    axes[0, 0].plot(
        1000 * waveform[window, 1],
        waveform[window, 2],
        linewidth=1,
        label="Quantiser input",
    )
    axes[0, 0].plot(
        1000 * waveform[window, 1],
        waveform[window, 4],
        ".",
        markersize=3,
        label=f"{bits}-bit samples",
    )
    axes[0, 0].set(
        xlabel="Time (ms)",
        ylabel="NPY amplitude units",
        title="Predetermined personal record c5/r0",
    )
    axes[0, 0].legend(fontsize=8)
    example_input = np.linspace(-4 * step, 4 * step, 601)
    indices = np.arange(2**bits)
    codebook = a * (indices / (2**bits - 1) * 2 - 1)
    midpoint = (codebook[:-1] + codebook[1:]) / 2
    analytic_output = codebook[
        np.searchsorted(midpoint, example_input - a * 1e-9, side="left")
    ]
    axes[0, 1].plot(
        example_input / step,
        analytic_output / step,
        drawstyle="steps-post",
        label="Codebook mapping",
    )
    axes[0, 1].plot(
        example_input / step, example_input / step, "--", color="grey"
    )
    axes[0, 1].set(
        xlabel="Input / step",
        ylabel="Output / step",
        title="Near zero: endpoint codebook has no zero level",
    )
    axes[1, 0].plot(
        1000 * waveform[:320, 1],
        waveform[:320, 5] / step,
        linewidth=0.7,
    )
    axes[1, 0].axhline(0.5, color="black", linestyle="--", linewidth=0.7)
    axes[1, 0].axhline(-0.5, color="black", linestyle="--", linewidth=0.7)
    axes[1, 0].set(
        xlabel="Time (ms)",
        ylabel="Error / step",
        title="Quantised minus input",
    )
    axes[1, 1].hist(
        waveform[:, 5] / step,
        bins=np.linspace(-0.5, 0.5, 31),
        density=True,
        alpha=0.7,
        label="32,000 observed errors",
    )
    axes[1, 1].axhline(
        1, color="black", linestyle="--", label="Uniform model density"
    )
    axes[1, 1].set(
        xlabel="Error / step",
        ylabel="Density",
        title="Model check; no claim of white/independent error",
    )
    axes[1, 1].legend(fontsize=8)
    figure.tight_layout()
    save_figure(figure, "r31_quantiser_errors")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for branch, label in [
        ("filtered", "Frozen filter"),
        ("unfiltered", "No filter"),
    ]:
        rows = sorted(
            [
                r
                for r in summary
                if r["branch"] == branch and int(r["bits"]) == bits
            ],
            key=lambda r: float(r["target_snr_db"]),
        )
        axes[0].plot(
            [float(r["target_snr_db"]) for r in rows],
            [float(r["mean_sqnr_db"]) for r in rows],
            "o-",
            label=label,
        )
    axes[0].set(
        xlabel="Target Gaussian SNR (dB)",
        ylabel="SQNR (dB)",
        title=f"Same {bits}-bit quantiser and fixed A={a:g}",
    )
    axes[0].legend(fontsize=8)
    example_rows = [
        r
        for r in trials
        if int(r["bits"]) == bits and r["branch"] == "filtered"
    ]
    delta = [100 * float(r["total_mse_change_fraction"]) for r in example_rows]
    component = [100 * float(r["component_ratio"]) for r in example_rows]
    axes[1].scatter(component, delta, s=18, label="40 design records")
    axes[1].plot(
        [0, max(component)],
        [0, max(component)],
        "--",
        color="grey",
        label="If cross term were zero",
    )
    axes[1].set(
        xlabel="Quantisation / upstream error power (%)",
        ylabel="Actual total MSE change (%)",
        title="Cross term retained on the full 48 kHz grid",
    )
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    save_figure(figure, "r31_range_fidelity")


if __name__ == "__main__":
    main()
