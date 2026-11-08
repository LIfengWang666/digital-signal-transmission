"""Regenerate the six final publication PDFs from measured project CSV.

The archived 29 October styling is preserved exactly. Paths are relative to
this package, and original scientific plots and measurements stay untouched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/supporting"
RESULT = ROOT / "stage6/results/publication_figures.json"
NAMES = (
    "r22_corrupted_spectral_evidence",
    "r22_parameter_search",
    "r22_response_boundary",
    "r23_recovery_quality",
    "r34_quality_gain_horizontal",
    "r34_bit_recovery_horizontal",
)
CAPTURED: dict[str, Figure] = {}
ORIGINAL_SAVE = Figure.savefig


class CapturedEnough(Exception):
    """Stop the plotting entry point after its two required R3.4 panels."""


def load_module(name: str, relative: str) -> ModuleType:
    """Load a supplied plotting module using its package-relative path.

    Args:
        name: Isolated import name for the plotting module.
        relative: Python source path relative to the unpacked package root.

    Returns:
        The executed module with its original CSV-only plotting functions.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plotting module: {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_save(
    figure: Figure, filename: str | Path, *args: Any, **kwargs: Any
) -> None:
    """Capture a figure instead of writing the original PNG/SVG output.

    Args:
        figure: Figure constructed by the original plotting function.
        filename: Original output path used to identify the figure stem.
        *args: Original save options, unused because no output is written.
        **kwargs: Original save options, unused because output is captured.

    Returns:
        None. Retain the figure object in the current export collection.
    """
    CAPTURED.setdefault(Path(filename).stem, figure)


def collect_original_figures() -> None:
    """Construct the six required panels with the original plotting code.

    Returns:
        None. Populate CAPTURED while suppressing original image writes.
    """
    CAPTURED.clear()
    with patch.object(Figure, "savefig", capture_save):
        r22 = load_module("publication_r22", "stage2/r22/plot_parameters.py")
        r22.main()
        r23 = load_module("publication_r23", "stage2/r23/plot_validation.py")
        r23.main()
        r34 = load_module("publication_r34", "stage3/r34/plot_validation.py")
        original_save = r34.save_figure
        count = 0

        def capture_horizontal(
            figure: Figure, name: str, horizontal: bool = False
        ) -> None:
            """Capture the required horizontal panels before manifest writes.

            Args:
                figure: R3.4 figure prepared from saved measurements.
                name: Original figure stem before the horizontal suffix.
                horizontal: Whether to apply the original horizontal layout.

            Returns:
                None. Raise CapturedEnough after the two required panels.
            """
            nonlocal count
            original_save(figure, name, horizontal)
            count += 1
            if count == 2:
                raise CapturedEnough()

        r34.save_figure = capture_horizontal
        with patch.object(r34.shutil, "copy2", return_value=None):
            try:
                r34.main(horizontal=True)
            except CapturedEnough:
                pass


def array_hash(value: Any) -> str:
    """Hash an array's shape, dtype and values for a plot-data record.

    Args:
        value: Array-like plotted coordinates, pixels or error-bar segments.

    Returns:
        A hexadecimal SHA-256 including shape and dtype.
    """
    data = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(data.shape).encode())
    h.update(str(data.dtype).encode())
    if data.dtype.hasobject:
        h.update(repr(data.tolist()).encode())
    else:
        h.update(data.tobytes())
    return h.hexdigest()


def science_fingerprint(figure: Figure) -> dict[str, Any]:
    """Describe plotted data and axes independently of visual formatting.

    Args:
        figure: Original or formatted Matplotlib figure to inspect.

    Returns:
        A canonical digest and per-axis coordinates, image and scale records.
    """
    axes = []
    for ax in figure.axes:
        item = {
            "xlim": list(ax.get_xlim()),
            "ylim": list(ax.get_ylim()),
            "xscale": ax.get_xscale(),
            "yscale": ax.get_yscale(),
            "lines": [
                {
                    "x": array_hash(line.get_xdata()),
                    "y": array_hash(line.get_ydata()),
                }
                for line in ax.lines
            ],
            "images": [
                {
                    "array": array_hash(im.get_array()),
                    "extent": list(im.get_extent()),
                    "clim": list(im.get_clim()),
                }
                for im in ax.images
            ],
            "collections": [],
            "patches": [array_hash(p.get_path().vertices) for p in ax.patches],
        }
        for collection in ax.collections:
            values = {
                "type": type(collection).__name__,
                "offsets": array_hash(collection.get_offsets()),
            }
            if isinstance(collection, LineCollection):
                values["segments"] = [
                    array_hash(s) for s in collection.get_segments()
                ]
            else:
                values["paths"] = [
                    array_hash(p.vertices) for p in collection.get_paths()
                ]
            if collection.get_array() is not None:
                values["array"] = array_hash(collection.get_array())
            item["collections"].append(values)
        axes.append(item)
    canonical = json.dumps(axes, sort_keys=True)
    return {
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "axes": axes,
    }


def set_text_and_lines(figure: Figure) -> None:
    """Apply the existing publication font, marker and line settings.

    Args:
        figure: Captured figure whose artist styles are changed in place.

    Returns:
        None. Preserve the existing publication appearance and plotted data.
    """
    # Font sizes are the actual PDF point sizes at the final LaTeX width.
    for text in figure.findobj(match=Text):
        text.set_fontfamily("DejaVu Sans")
        text.set_fontsize(8.0)
    for ax in figure.axes:
        ax.xaxis.label.set_fontsize(8.7)
        ax.yaxis.label.set_fontsize(8.7)
        ax.title.set_fontsize(8.4)
        ax.set_title(ax.get_title(), pad=5, fontsize=8.4)
        ax.tick_params(
            axis="both", which="both", labelsize=8, width=0.6, pad=2.3
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        for line in ax.lines:
            line.set_linewidth(1.05)
            # Markers must not dominate a panel at its printed size.
            if line.get_marker() not in (None, "", "None", " "):
                if line.get_marker() == "*":
                    line.set_markersize(6.5)
                else:
                    line.set_markersize(min(line.get_markersize(), 3.4))
                line.set_markeredgewidth(0.65)
        for collection in ax.collections:
            if isinstance(collection, LineCollection):
                collection.set_linewidth(0.9)
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            gridline.set_linewidth(0.4)
            gridline.set_color("#d9d9d9")
            gridline.set_alpha(0.75)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(8.0)
            legend.set_frame_on(False)
        ax.xaxis.get_offset_text().set_fontsize(8)
        ax.yaxis.get_offset_text().set_fontsize(8)


def legend(ax: Axes, *, loc: str = "best", **kwargs: Any) -> Legend:
    """Create a legend using the existing publication settings.

    Args:
        ax: Axes supplying the legend entries.
        loc: Matplotlib legend placement name.
        **kwargs: Additional legend options for the existing panel layout.

    Returns:
        The created Matplotlib legend.
    """
    return ax.legend(
        loc=loc,
        fontsize=8,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.4,
        borderaxespad=0.25,
        labelspacing=0.25,
        **kwargs,
    )


def style_figure(stem: str, figure: Figure) -> None:
    """Restore the exact existing publication layout for one figure.

    Args:
        stem: One of the six registered scientific figure stems.
        figure: Captured plot to format at its existing physical size.

    Returns:
        None. Change presentation in place without altering plotted data.
    """
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    width_mm = 139.32 if stem.startswith("r34") else 132.84
    old_w, old_h = figure.get_size_inches()
    figure.set_size_inches(width_mm / 25.4, width_mm / 25.4 * old_h / old_w)
    set_text_and_lines(figure)
    axes = figure.axes
    if stem == "r22_corrupted_spectral_evidence":
        figure.set_size_inches(width_mm / 25.4, 96 / 25.4)
        axes[0].set_title(
            "Measured corrupted spectrum\nID baseline, design repeat 0"
        )
        axes[0].set_ylabel("PSD (NPY units²/Hz)")
        axes[1].set_xlabel("Frequency offset (Hz)")
        axes[2].set_xlabel("Frequency offset (Hz)")
        axes[1].set_ylabel("Amplitude (dB re peak)")
        axes[2].set_ylabel("Amplitude (dB re peak)")
        # Dense spectrum is a trace rather than one of the smooth main curves.
        axes[0].lines[0].set_linewidth(0.6)
        overview_handles, overview_labels = axes[0].get_legend_handles_labels()
        window_handles, window_labels = axes[1].get_legend_handles_labels()
        for ax in axes:
            ax.get_legend().remove()
        # Both legends occupy their own horizontal rows rather than covering
        # the measured noisy spectrum or the low-amplitude window side lobes.
        axes[0].set_position((0.13, 0.64, 0.84, 0.25))
        axes[1].set_position((0.13, 0.15, 0.36, 0.28))
        axes[2].set_position((0.61, 0.15, 0.36, 0.28))
        figure.legend(
            overview_handles,
            overview_labels,
            loc="upper center",
            bbox_to_anchor=(0.55, 0.55),
            ncol=2,
            fontsize=8,
            frameon=False,
            handlelength=1.8,
            columnspacing=1.2,
        )
        figure.legend(
            window_handles,
            window_labels,
            loc="lower center",
            bbox_to_anchor=(0.55, 0.005),
            ncol=3,
            fontsize=8,
            frameon=False,
            handlelength=1.8,
            columnspacing=1.2,
        )
    elif stem == "r22_parameter_search":
        figure.set_size_inches(width_mm / 25.4, 72 / 25.4)
        axes[0].set_title("Hann refinement\nWhite cells fail gain constraint")
        axes[1].set_title("Window/length tradeoff\nWithin the tested grid")
        # Original axes order: first heatmap, right curve, final colorbar.
        axes[0].set_xlabel("Half-width (Hz)")
        axes[1].set_ylabel("Best feasible score J")
        for text in axes[0].get_xticklabels() + axes[1].get_xticklabels():
            text.set_rotation(90 if text in axes[0].get_xticklabels() else 45)
            text.set_horizontalalignment(
                "center" if text in axes[0].get_xticklabels() else "right"
            )
        legend(axes[1], loc="upper left")
        figure.tight_layout(pad=0.5, w_pad=1.0)
    elif stem == "r22_response_boundary":
        figure.set_size_inches(width_mm / 25.4, 92 / 25.4)
        for ax in axes[:2]:
            ax.set_ylabel("Magnitude (dB re unity)")
        axes[0].set_title("All four passbands")
        axes[1].set_title("Retained 6 kHz: detail")
        axes[2].set_title("Beginning of clean record")
        axes[3].set_title("End of clean record")
        for ax in axes[2:]:
            ax.set_ylabel("Clean-input error\n(NPY units)")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].get_legend().remove()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=1,
            fontsize=8,
            frameon=False,
            handlelength=1.8,
            labelspacing=0.25,
            bbox_to_anchor=(0.5, 1.0),
        )
        for ax in axes[2:]:
            legend(ax, loc="upper left" if ax is axes[3] else "upper right")
        figure.tight_layout(
            pad=0.5, w_pad=1.3, h_pad=1.5, rect=(0, 0, 1, 0.89)
        )
    elif stem == "r23_recovery_quality":
        figure.set_size_inches(width_mm / 25.4, 105 / 25.4)
        axes[0].set_title(
            "New phase-1 noise\nFixed coefficients and boundaries"
        )
        axes[0].set_ylabel("48 kHz full-record MSE\n(NPY units²)")
        axes[1].set_xlabel("Target sample SNR (dB)")
        axes[1].set_ylabel("Paired error-SNR gain (dB)")
        legend(axes[0], loc="upper right")
        legend(axes[1], loc="lower left")
        # Reflow two explanatory labels without changing their data anchors.
        for text in axes[1].texts:
            if text.get_text().startswith("Sample-domain"):
                text.set_text("Sample-domain crossover\n34.79 dB")
                text.set_position((7.0, 3.8))
                text.set_horizontalalignment("left")
            elif text.get_text().startswith("ID baseline"):
                text.set_text("ID baseline\n2.661 dB")
                text.set_position((7.2, 8.8))
        figure.tight_layout(pad=0.5, h_pad=1.1)
    elif stem == "r34_quality_gain_horizontal":
        figure.set_size_inches(width_mm / 25.4, 68 / 25.4)
        axes[0].set_title(
            "300 new input records\nAdjusted intervals over 30 means"
        )
        axes[1].set_title("Transition detail\nGreen: pass; red: fail")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].get_legend().remove()
        # Put a longer threshold label in a page-width legend rather than
        # reducing its font until it fits a narrow scientific panel.
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            fontsize=8,
            frameon=False,
            handlelength=1.5,
            columnspacing=0.9,
            handletextpad=0.4,
            bbox_to_anchor=(0.5, 1.0),
        )
        legend(axes[1], loc="upper left")
        figure.tight_layout(pad=0.5, w_pad=1.2, rect=(0, 0, 1, 0.9))
    elif stem == "r34_bit_recovery_horizontal":
        figure.set_size_inches(width_mm / 25.4, 69 / 25.4)
        axes[0].set_title(
            "Exact decoder expectations\nDots: validation counts"
        )
        axes[1].set_title(
            "Whole-payload recovery\nPointwise exact 95% intervals"
        )
        axes[1].set_ylabel("Bit-perfect payload fraction")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(
            handles,
            ["Uncoded: exact", "Hamming (7,4): exact"],
            loc="lower right",
            fontsize=8,
            frameon=False,
            handlelength=1.8,
            handletextpad=0.4,
            labelspacing=0.25,
        )
        legend(axes[1], loc="upper right")
        figure.tight_layout(pad=0.5, w_pad=1.2)
    else:
        raise ValueError(stem)
    # Explicit titles above used set_title, which otherwise restores global rc
    # sizes from the last original module. Finalize actual printed sizes here.
    for ax in axes:
        ax.title.set_fontsize(8.4)
        ax.xaxis.label.set_fontsize(8.7)
        ax.yaxis.label.set_fontsize(8.7)
    figure.canvas.draw()


def file_hashes() -> dict[str, str]:
    """Snapshot scientific sources and existing numerical/image products.

    Returns:
        Package-relative SHA-256 values, excluding caches and this export's
        own result record. Publication PDFs are checked separately.
    """
    extensions = {
        ".py",
        ".csv",
        ".json",
        ".png",
        ".svg",
        ".npy",
        ".bin",
        ".wav",
    }
    values = {}
    for current, directories, files in os.walk(ROOT):
        directories[:] = [
            name
            for name in directories
            if name not in {".work", ".build-cache", "__pycache__", ".git"}
        ]
        for name in files:
            p = Path(current) / name
            if p.suffix in extensions and p != RESULT:
                values[p.relative_to(ROOT).as_posix()] = hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
    return values


def main() -> None:
    """Regenerate six final PDFs and record data/presentation integrity.

    Returns:
        None. Write the six report assets and an export verification JSON;
        raise if the styling changes data or existing scientific products.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    protected_before = file_hashes()
    collect_original_figures()
    records = []
    for stem in NAMES:
        figure = CAPTURED[stem]
        before = science_fingerprint(figure)
        style_figure(stem, figure)
        after = science_fingerprint(figure)
        if before["sha256"] != after["sha256"]:
            raise AssertionError(f"Scientific artist data changed: {stem}")
        destination = OUT / ("publication_" + stem + ".pdf")
        ORIGINAL_SAVE(
            figure,
            destination,
            format="pdf",
            metadata={
                "CreationDate": None,
                "ModDate": None,
                "Creator": "Matplotlib; presentation-only export",
            },
        )
        records.append(
            {
                "stem": stem,
                "path": destination.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "canvas_mm": (figure.get_size_inches() * 25.4).tolist(),
                "before_data_sha256": before["sha256"],
                "after_data_sha256": after["sha256"],
                "scientific_artist_data_equal": True,
            }
        )
    plt.close("all")
    if protected_before != file_hashes():
        raise AssertionError("An original source/data/image file changed")
    result = {
        "status": "passed",
        "purpose": "Six final publication PDFs",
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "all_original_sources_data_and_PNG_SVG_unchanged": True,
        "protected_file_count": len(protected_before),
        "new_independent_observations": 0,
        "font_sizes_pt": {
            "axis_labels": 8.7,
            "ticks_and_legend": 8.0,
            "titles": 8.4,
        },
        "records": records,
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Six final publication PDFs regenerated; plotted data unchanged.")


if __name__ == "__main__":
    main()
