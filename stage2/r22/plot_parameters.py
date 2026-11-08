"""Plot the R2.2 parameter search and the selected filter's boundary cost."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from stage2.r22.filter_parameters import read_csv  # noqa: E402


def plot_spectral_evidence(results: Path, figures: Path) -> None:
    """Expose the measured noisy PSD and analysis-window peak widths.

    Args:
        results: Existing R2.2 CSV directory.
        figures: Figure destination.

    Returns:
        None. Save PNG/SVG without changing any filter or random stream.
    """
    native = [
        row
        for row in read_csv(results / "corrupted_spectrum.csv")
        if row["window"] == "rectangular"
    ]
    local = read_csv(results / "local_peak_spectra.csv")
    evidence = read_csv(results / "spectral_evidence.csv")
    background = float(evidence[0]["offband_mean_psd"])
    frequency = np.array([float(row["frequency_hz"]) for row in native])
    psd = np.array([float(row["psd"]) for row in native])
    fig = plt.figure(figsize=(11, 7.5))
    layout = fig.add_gridspec(2, 2)
    overview = fig.add_subplot(layout[0, :])
    overview.semilogy(
        frequency, psd, lw=0.65, color="tab:orange", label="Corrupted PSD"
    )
    overview.axhline(
        background,
        color="black",
        ls="--",
        lw=1,
        label="Measured off-band mean",
    )
    for row in evidence:
        centre = float(row["centre_hz"])
        peak = float(row["rectangular_peak_psd"])
        overview.plot(centre, peak, "o", color="tab:blue", ms=4)
        overview.annotate(
            f"{centre:g} Hz",
            (centre, peak),
            xytext=(3, 7),
            textcoords="offset points",
            fontsize=11,
        )
    overview.set(
        xlim=(0, 8000),
        ylim=(1e-9, 5),
        xlabel="Frequency (Hz)",
        ylabel="PSD (NPY units squared / Hz)",
        title="Measured corrupted spectrum: ID baseline, design repeat 0",
    )
    overview.legend(fontsize=11, loc="upper right")
    for column, centre in enumerate([440.0, 6000.0]):
        ax = fig.add_subplot(layout[1, column])
        for window, label in [
            ("rectangular", "Rectangular"),
            ("hann", "Hann"),
            ("hamming", "Hamming"),
        ]:
            rows = [
                row
                for row in local
                if row["window"] == window
                and float(row["centre_hz"]) == centre
            ]
            offsets = np.array([float(row["offset_hz"]) for row in rows])
            amplitude = np.array([float(row["amplitude"]) for row in rows])
            local_peak = np.max(amplitude[np.abs(offsets) <= 1])
            relative_db = 20 * np.log10(
                np.maximum(amplitude / local_peak, 1e-8)
            )
            ax.plot(offsets, relative_db, label=label, lw=1.2)
        ax.axhline(-3, color="gray", ls=":", lw=1)
        ax.set(
            xlim=(-2, 2),
            ylim=(-45, 1),
            xlabel="Frequency offset from centre (Hz)",
            ylabel="Amplitude (dB re local peak)",
            title=f"Analysis peak at {centre:g} Hz",
        )
        ax.legend(fontsize=11, loc="lower left")
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(
            figures / f"r22_corrupted_spectral_evidence.{suffix}", dpi=300
        )
    plt.close(fig)


def main() -> None:
    """Read saved CSV and show selection, response and endpoint distortion.

    Returns:
        None. Save PNG/SVG figures without recomputing the parameter scan.
    """
    results = ROOT / "results"
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(
        (results / "filter_freeze.json").read_text(encoding="utf-8")
    )
    selected = frozen["selected"]
    scan = read_csv(results / "parameter_scan.csv")
    plt.rcdefaults()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    lengths = frozen["refined_lengths"]
    widths = frozen["refined_half_widths_hz"]
    grid = np.full((len(lengths), len(widths)), np.nan)
    for row in scan:
        length = int(row["length"])
        width = float(row["half_width_hz"])
        if (
            row["window"] == "hann"
            and row["feasible"] == "True"
            and length in lengths
            and width in widths
        ):
            grid[lengths.index(length), widths.index(width)] = float(
                row["score"]
            )
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    mesh = axs[0].imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        vmin=np.nanmin(grid),
        vmax=np.nanmax(grid),
    )
    axs[0].set_xticks(
        range(len(widths)), [f"{v:g}" for v in widths], rotation=45
    )
    axs[0].set_yticks(range(len(lengths)), lengths)
    axs[0].plot(
        widths.index(selected["half_width_hz"]),
        lengths.index(selected["length"]),
        "*",
        color="red",
        markersize=13,
        markeredgecolor="white",
    )
    axs[0].set(
        xlabel="Nominal half-width (Hz)",
        ylabel="Kernel length M",
        title="Hann refinement\nWhite cells fail the gain constraint",
    )
    axs[0].grid(False)
    fig.colorbar(mesh, ax=axs[0], label="Score J (lower is better)")
    for window, label in [
        ("rectangular", "Rectangular"),
        ("hann", "Hann"),
        ("hamming", "Hamming"),
    ]:
        rows = [
            row
            for row in scan
            if row["window"] == window and row["feasible"] == "True"
        ]
        tested_lengths = sorted({int(row["length"]) for row in rows})
        scores = [
            min(
                float(row["score"])
                for row in rows
                if int(row["length"]) == length
            )
            for length in tested_lengths
        ]
        axs[1].plot(tested_lengths, scores, "o-", label=label, markersize=4)
    axs[1].plot(
        selected["length"],
        selected["score"],
        "*",
        color="red",
        markersize=13,
        markeredgecolor="white",
    )
    axs[1].set(
        xscale="log",
        xlabel="Kernel length M",
        ylabel="Best feasible score J",
        title="Window/length tradeoff\nWithin the tested grid",
    )
    length_ticks = [257, 385, 513, 1025, 2049, 4097]
    axs[1].set_xticks(length_ticks, length_ticks, rotation=45)
    axs[1].legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(figures / "r22_parameter_search.png", dpi=300)
    fig.savefig(figures / "r22_parameter_search.svg")
    plt.close(fig)

    response = read_csv(results / "selected_response.csv")
    prototype = [
        row
        for row in read_csv(
            PACKAGE / "stage2/r21/results/prototype_response.csv"
        )
        if row["method"] == "multiband"
    ]
    f = np.array([float(row["frequency_hz"]) for row in response])
    gain = np.array([float(row["magnitude"]) for row in response])
    old_gain = np.array([float(row["magnitude"]) for row in prototype])
    boundary = read_csv(results / "boundary_records.csv")
    t = 1000 * np.array([float(row["time_s"]) for row in boundary])
    error = np.array([float(row["selected_clean_error"]) for row in boundary])
    old_error = np.array(
        [float(row["prototype_clean_error"]) for row in boundary]
    )
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for ax, limits in zip(axs[0], [(0, 8000), (5800, 6200)]):
        for values, label in [
            (old_gain, "R2.1: 513 taps, +/-80 Hz, Hann"),
            (
                gain,
                f"Selected: {selected['length']} taps, "
                f"+/-{selected['half_width_hz']:g} Hz, "
                f"{selected['window'].title()}",
            ),
        ]:
            ax.plot(
                f,
                20 * np.log10(np.maximum(values, 1e-5)),
                label=label,
                linewidth=1,
            )
        ax.set(
            xlim=limits,
            ylim=(-70, 2),
            xlabel="Frequency (Hz)",
            ylabel="Filter magnitude (dB re unity)",
        )
    axs[0, 0].set_title("All four realised passbands")
    axs[0, 0].legend(fontsize=11, loc="lower left")
    axs[0, 1].set_title("Retained 6 kHz: passband detail")
    detail_centre = max(frozen["centres_hz"])
    for value in [
        detail_centre - selected["half_width_hz"],
        detail_centre + selected["half_width_hz"],
    ]:
        axs[0, 1].axvline(value, color="grey", linestyle=":", linewidth=0.8)
    for ax, mask, title in [
        (axs[1, 0], t <= 25, "Beginning of the clean record"),
        (axs[1, 1], t >= 1975, "End of the clean record"),
    ]:
        ax.plot(t[mask], old_error[mask], label="R2.1 prototype", linewidth=1)
        ax.plot(t[mask], error[mask], label="Selected filter", linewidth=1)
        ax.set(
            xlabel="Time (ms)",
            ylabel="Clean-input error (NPY units)",
            title=title,
        )
        ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(figures / "r22_response_boundary.png", dpi=300)
    fig.savefig(figures / "r22_response_boundary.svg")
    plt.close(fig)
    plot_spectral_evidence(results, figures)
    print("R2.2 figures generated from saved CSV.")


if __name__ == "__main__":
    main()
