"""Compare sufficient sampling with intentional unprotected undersampling.

The input is the original NPY file. Sampling, spectra and reconstruction
use no noise, quantisation, random seed or source-tone-assisted recovery.
Run from the package root: python -m stage1.r13.aliasing_experiment.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from stage1.r11.characterise_reference import (
    FS_HZ,
    THRESHOLD_DB,
    component_peaks,
    spectrum,
)
from stage1.r12.sampling_rate import check_row, periodic_reconstruct

matplotlib.use("Agg")

FloatArray = NDArray[np.float64]
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
UNDER_FACTOR = 6


def write_csv(name: str, rows: list[dict]) -> None:
    """Save measured rows beneath the local results directory.

    Args:
        name: Output CSV filename.
        rows: Nonempty dictionaries with common field names.

    Returns:
        None. Write a header and all rows without rounding values.
    """
    with (ROOT / "results" / name).open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def alias_predictions(components: list[dict], fs_hz: float) -> list[dict]:
    """Predict signed folding and cosine phase from reference sampling identities.

    Args:
        components: R1.1 component frequency, amplitude and phase records.
        fs_hz: Positive sampling frequency in Hz.

    Returns:
        Per-source predictions of principal frequency and cosine phase.
        DC and Nyquist coincidences are excluded from this experiment.
    """
    rows = []
    for component in components:
        source = float(component["frequency_hz"])
        amplitude = float(component["amplitude_npy_units"])
        phase = float(component["phase_cosine_rad"])
        signed = (source + fs_hz / 2) % fs_hz - fs_hz / 2
        alias = abs(signed)
        if not 0 < alias < fs_hz / 2:
            raise ValueError("This experiment requires interior components.")
        rows.append(
            {
                "source_frequency_hz": source,
                "source_amplitude": amplitude,
                "source_phase_rad": phase,
                "sampling_multiple": int(round((source - signed) / fs_hz)),
                "signed_frequency_hz": signed,
                "predicted_frequency_hz": alias,
                "predicted_phase_rad": -phase if signed < 0 else phase,
            }
        )
    return rows


def predicted_mse(predictions: list[dict]) -> float:
    """Predict error power using paired complex coefficients and orthogonality.

    Args:
        predictions: Source and aliased component records whose frequencies
            are integer multiples of 1/T on the same declared period T.

    Returns:
        Mean-square difference between the coherent periodic models.
        This is explanatory theory, never an input to reconstruction.
    """
    original: dict[float, complex] = {}
    recovered: dict[float, complex] = {}
    for row in predictions:
        source = row["source_frequency_hz"]
        folded = row["predicted_frequency_hz"]
        amplitude = row["source_amplitude"]
        original[source] = original.get(source, 0j) + amplitude / 2 * np.exp(
            1j * row["source_phase_rad"]
        )
        recovered[folded] = recovered.get(folded, 0j) + amplitude / 2 * np.exp(
            1j * row["predicted_phase_rad"]
        )
    return float(
        2
        * sum(
            abs(recovered.get(f, 0j) - original.get(f, 0j)) ** 2
            for f in original.keys() | recovered.keys()
        )
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    """Export a figure in both required viewing and vector formats.

    Args:
        fig: Matplotlib figure ready for export.
        name: Output filename stem beneath the figures directory.

    Returns:
        None. Save 300 dpi PNG and SVG files, then close the figure.
    """
    fig.savefig(ROOT / "figures" / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_experiment(
    reference: FloatArray, cases: list[dict], band_edge: float
) -> None:
    """Plot measured spectra and corresponding waveform/error evidence.

    Args:
        reference: Original reference at FS_HZ.
        cases: Measured samples, spectra and reconstructions for both paths.
        band_edge: R1.1 highest identified source frequency in Hz.

    Returns:
        None. Save two PNG/SVG figure pairs.
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    base, under = cases
    max_hz = base["fs_hz"] / 2
    f_ref, _, a_ref = spectrum(reference, FS_HZ, np.ones(reference.size))
    level = np.max(a_ref) * 10 ** (THRESHOLD_DB / 20)
    fig, axes = plt.subplots(
        3, 1, figsize=(9, 7.4), sharex=True, layout="constrained"
    )
    plots = [(f_ref, a_ref, "#777777", "Original 48 kHz reference")]
    plots += [
        (
            c["frequencies"],
            c["amplitudes"],
            color,
            f"Measured {c['fs_hz']/1000:g} kHz samples",
        )
        for c, color in zip(cases, ["#225d91", "#b45b4c"])
    ]
    for ax, (frequencies, amplitudes, color, label) in zip(axes, plots):
        keep = (amplitudes >= level) & (frequencies <= max_hz)
        ax.vlines(frequencies[keep] / 1000, 0, amplitudes[keep], color=color)
        ax.scatter(
            frequencies[keep] / 1000, amplitudes[keep], s=22, color=color
        )
        ax.set(ylabel="Amplitude (NPY units)", ylim=(0, 1.14), title=label)
        ax.grid(alpha=0.2)
    nyquist = under["fs_hz"] / 2
    folded_hz = abs((band_edge + nyquist) % under["fs_hz"] - nyquist)
    folded_bin = int(round(folded_hz * reference.size / FS_HZ))
    folded_amplitude = under["amplitudes"][folded_bin]
    axes[2].axvspan(nyquist / 1000, max_hz / 1000, color="#eeeeee")
    axes[2].axvline(nyquist / 1000, ls="--", color="#555555")
    axes[2].text(
        0.98,
        0.88,
        f"Shaded: above the {under['fs_hz']/1000:g} kHz path's Nyquist limit",
        transform=axes[2].transAxes,
        ha="right",
        fontsize=8,
    )
    axes[2].annotate(
        f"{band_edge:g} -> {folded_hz:g} Hz",
        (folded_hz / 1000, folded_amplitude),
        xytext=(folded_hz / 1000 - 0.4, 0.9),
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": "#b45b4c"},
    )
    axes[2].annotate(
        "Original 2.2 kHz component",
        (2.2, 0.3),
        xytext=(2.6, 0.65),
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": "#b45b4c"},
    )
    axes[2].set(xlabel="Frequency (kHz)", xlim=(0, max_hz / 1000))
    save_figure(fig, "aliasing_spectra")

    t = np.arange(reference.size) / FS_HZ
    visible = t < 0.004
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.6), layout="constrained")
    axes[0].plot(
        t[visible] * 1000,
        reference[visible],
        color="#777777",
        lw=1.8,
        label="Original reference",
    )
    for c, color in zip(cases, ["#225d91", "#b45b4c"]):
        label = f"{c['fs_hz']/1000:g} kHz reconstruction"
        axes[0].plot(
            t[visible] * 1000,
            c["recovered"][visible],
            color=color,
            ls="--" if c is base else "-",
            lw=1.1,
            label=label,
        )
        axes[1].plot(
            t[visible] * 1000,
            c["error"][visible],
            color=color,
            lw=1.1,
            label=label,
        )
    axes[0].set(
        ylabel="Amplitude (NPY units)",
        xlabel="Time (ms)",
        title="First 4 ms; metrics use the entire 2 s",
    )
    low_t = np.arange(under["samples"].size) / under["fs_hz"]
    low_visible = low_t < 0.004
    axes[0].scatter(
        low_t[low_visible] * 1000,
        under["samples"][low_visible],
        facecolors="white",
        edgecolors="#b45b4c",
        s=18,
        label="8 kHz input samples",
        zorder=4,
    )
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].set(
        ylabel="Error (NPY units)",
        xlabel="Time (ms)",
        title="Error is reconstruction minus the original reference",
    )
    error_f, _, error_a = spectrum(
        under["error"], FS_HZ, np.ones(reference.size)
    )
    keep = (error_a >= level) & (error_f <= max_hz)
    axes[2].vlines(error_f[keep] / 1000, 0, error_a[keep], color="#b45b4c")
    axes[2].scatter(error_f[keep] / 1000, error_a[keep], color="#b45b4c")
    axes[2].set(
        xlabel="Frequency (kHz)",
        ylabel="Error amplitude (NPY units)",
        xlim=(0, max_hz / 1000),
        ylim=(0, 0.12),
        title=("Full-record error spectrum: " "added alias and lost tone"),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    save_figure(fig, "aliasing_reconstruction_error")


def main() -> None:
    """Read the reference/settings and run the controlled comparison.

    Returns:
        None. Write full CSV, figures, predictions and verification results.
    """
    for directory in ["results", "figures"]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    path = PROJECT / "reference_signal.npy"
    input_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    reference = np.load(path, allow_pickle=False)
    r11 = ROOT.parent / "r11" / "results"
    with (r11 / "components.csv").open(encoding="utf-8") as stream:
        components = list(csv.DictReader(stream))
    with (ROOT.parent / "r12/results/summary.csv").open(
        encoding="utf-8"
    ) as stream:
        settings = {
            r["metric"]: float(r["value"]) for r in csv.DictReader(stream)
        }
    source_metadata = json.loads((r11 / "metadata.json").read_text())
    if input_hash != source_metadata["input_npy_sha256"]:
        raise ValueError("R1.1 predictions refer to a different input file.")
    band_edge = settings["measured_band_edge_hz"]
    base_factor = int(settings["selected_factor"])
    if not FS_HZ / UNDER_FACTOR < 2 * band_edge < FS_HZ / base_factor:
        raise ValueError("A strictly insufficient/sufficient pair is needed.")
    cases, metrics, mapping, spectra_rows, sample_rows, checks = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    duration = reference.size / FS_HZ
    reference_amplitude = max(
        float(row["amplitude_npy_units"]) for row in components
    )
    for factor in [base_factor, UNDER_FACTOR]:
        if reference.size % factor:
            raise ValueError("Each path must retain the same complete period.")
        fs_hz = FS_HZ / factor
        samples = reference[::factor].copy()
        predictions = alias_predictions(components, fs_hz)
        expected_f = [p["predicted_frequency_hz"] for p in predictions]
        if len(set(expected_f)) != len(expected_f):
            raise ValueError(
                "This selected comparison has no peak collisions."
            )
        f, _, amplitude = spectrum(samples, fs_hz, np.ones(samples.size))
        Y = np.fft.rfft(samples)
        peaks = component_peaks(amplitude, reference_amplitude, THRESHOLD_DB)
        recovered = periodic_reconstruct(samples, reference.size)
        error = recovered - reference
        model_samples = np.zeros(samples.size)
        maximum_coefficient_error = 0.0
        maximum_phase_error = 0.0
        for prediction in predictions:
            k = int(
                peaks[
                    np.argmin(
                        abs(f[peaks] - prediction["predicted_frequency_hz"])
                    )
                ]
            )
            theory_phase = prediction["predicted_phase_rad"]
            expected_c = (
                prediction["source_amplitude"] / 2 * np.exp(1j * theory_phase)
            )
            coefficient_error = float(abs(Y[k] / samples.size - expected_c))
            phase_error = float(
                abs(np.angle(np.exp(1j * (np.angle(Y[k]) - theory_phase))))
            )
            maximum_coefficient_error = max(
                maximum_coefficient_error, coefficient_error
            )
            maximum_phase_error = max(maximum_phase_error, phase_error)
            model_samples += prediction["source_amplitude"] * np.cos(
                2
                * np.pi
                * prediction["predicted_frequency_hz"]
                * np.arange(samples.size)
                / fs_hz
                + theory_phase
            )
            mapping.append(
                {
                    "sample_rate_hz": fs_hz,
                    **prediction,
                    "measured_frequency_hz": float(f[k]),
                    "measured_amplitude": float(amplitude[k]),
                    "measured_phase_rad": float(np.angle(Y[k])),
                    "coefficient_absolute_error": coefficient_error,
                    "wrapped_phase_error_rad": phase_error,
                }
            )
        mse = float(np.mean(error**2))
        theory_mse = predicted_mse(predictions)
        row = {
            "sample_rate_hz": fs_hz,
            "factor": factor,
            "sample_count": samples.size,
            "duration_s": samples.size / fs_hz,
            "last_sample_time_s": (samples.size - 1) / fs_hz,
            "nyquist_frequency_hz": fs_hz / 2,
            "strict_nyquist": bool(fs_hz > 2 * band_edge),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "nmse": mse / float(np.mean(reference**2)),
            "maximum_absolute_error": float(np.max(abs(error))),
            "sample_consistency_error": float(
                np.max(abs(recovered[::factor] - samples))
            ),
            "predicted_mse": theory_mse,
        }
        metrics.append(row)
        prefix = f"{fs_hz:g} Hz: "
        actual_f = sorted(f[peaks].tolist())
        predicted_f = sorted(expected_f)
        checks.extend(
            [
                check_row(
                    prefix + "same complete duration",
                    abs(samples.size / fs_hz - duration),
                    1e-15,
                ),
                check_row(
                    prefix + "detected peak count",
                    abs(len(actual_f) - len(predicted_f)),
                    0,
                ),
                check_row(
                    prefix + "predicted versus detected frequencies",
                    float(np.max(abs(np.array(actual_f) - predicted_f))),
                    1e-9,
                ),
                check_row(
                    prefix + "predicted complex coefficients",
                    maximum_coefficient_error,
                    1e-12,
                ),
                check_row(
                    prefix + "predicted reflected phases",
                    maximum_phase_error,
                    1e-10,
                ),
                check_row(
                    prefix + "alias model agrees on sample grid",
                    float(np.max(abs(samples - model_samples))),
                    1e-10,
                ),
                check_row(
                    prefix + "reconstruction fits received samples",
                    row["sample_consistency_error"],
                    1e-12,
                ),
                check_row(
                    prefix + "MSE agrees with spectral error prediction",
                    abs(mse - theory_mse),
                    1e-12,
                ),
            ]
        )
        cases.append(
            {
                "fs_hz": fs_hz,
                "samples": samples,
                "frequencies": f,
                "amplitudes": amplitude,
                "recovered": recovered,
                "error": error,
            }
        )
        spectra_rows.extend(
            {
                "sample_rate_hz": fs_hz,
                "frequency_hz": float(freq),
                "amplitude": float(value),
            }
            for freq, value in zip(f, amplitude)
        )
        sample_rows.extend(
            {
                "sample_rate_hz": fs_hz,
                "index": m,
                "time_s": m / fs_hz,
                "value": float(value),
            }
            for m, value in enumerate(samples)
        )
    base, under = cases
    error_f, _, error_a = spectrum(
        under["error"], FS_HZ, np.ones(reference.size)
    )
    write_csv("comparison.csv", metrics)
    write_csv("alias_mapping.csv", mapping)
    write_csv("spectra.csv", spectra_rows)
    write_csv("low_rate_samples.csv", sample_rows)
    write_csv(
        "error_spectrum.csv",
        [
            {"frequency_hz": float(f), "amplitude": float(a)}
            for f, a in zip(error_f, error_a)
        ],
    )
    write_csv(
        "reconstructions.csv",
        [
            {
                "index": n,
                "time_s": n / FS_HZ,
                "reference": float(x),
                "baseline_reconstruction": float(base["recovered"][n]),
                "undersampled_reconstruction": float(under["recovered"][n]),
                "baseline_error": float(base["error"][n]),
                "undersampled_error": float(under["error"][n]),
            }
            for n, x in enumerate(reference)
        ],
    )
    units = [
        "s",
        "count",
        "Hz",
        "NPY units",
        "rad",
        "NPY units",
        "NPY units",
        "NPY units squared",
    ] * len(cases)
    for row, unit in zip(checks, units, strict=True):
        row["unit"] = unit
    write_csv("checks.csv", checks)
    summary = {
        "baseline_fs_hz": base["fs_hz"],
        "under_fs_hz": under["fs_hz"],
        "band_edge_hz": band_edge,
        "duration_s": duration,
        "evaluation_points": reference.size,
        "baseline_mse": metrics[0]["mse"],
        "under_mse": metrics[1]["mse"],
        "under_rmse": metrics[1]["rmse"],
        "under_nmse": metrics[1]["nmse"],
        "under_maximum_error": metrics[1]["maximum_absolute_error"],
        "predicted_under_mse": metrics[1]["predicted_mse"],
        "deterministic_check_count": len(checks),
    }
    write_csv(
        "summary.csv", [{"metric": k, "value": v} for k, v in summary.items()]
    )
    metadata = {
        "input_npy_sha256": input_hash,
        "conditions": "Clean reference, direct index selection; no noise, "
        "quantisation or bit errors.",
        "anti_alias_filter": "Intentionally bypassed to expose folding; "
        "not the recommendation for a general sampler.",
        "reconstruction": "Same R1.2 periodic Fourier interpolation for "
        "both paths; only low-rate samples enter recovery.",
        "evaluation": "All original grid positions; no trimming, fitted "
        "gain, time shift or original-tone-assisted recovery.",
        "prediction_only": "Source parameters explain the expected folds "
        "and error power after sampling; they are never "
        "passed to the recovery function.",
        "random_seed": None,
    }
    (ROOT / "results/metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_experiment(reference, cases, band_edge)
    if hashlib.sha256(path.read_bytes()).hexdigest() != input_hash:
        raise ArithmeticError("The original reference changed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
