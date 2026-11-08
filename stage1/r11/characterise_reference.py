"""Reproduce the R1.1 spectrum, window decision, and operational band edge.

Uses the supplied NPY amplitudes unchanged. The 48 kHz rate comes from the
project and is cross-checked against WAV metadata. No degradation, sampling
rate selection, or low-rate reconstruction is performed here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import wave
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IndexArray = NDArray[np.int64]
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
FS_HZ = 48_000.0
THRESHOLD_DB = -60.0
SENSITIVITY_DB = (-40.0, -60.0, -80.0)


def save_csv(name: str, rows: list[dict]) -> None:
    """Write dictionaries rows to results/name with a header; return None.

    Args:
        name: Output filename beneath this step's results directory.
        rows: Nonempty result dictionaries sharing the same field names.

    Returns:
        None. Write a header and all measured rows.
    """
    with (ROOT / "results" / name).open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spectrum(
    samples: FloatArray, fs_hz: float, window: FloatArray
) -> tuple[FloatArray, ComplexArray, FloatArray]:
    """Return Hz bins, raw rDFT and one-sided amplitudes for
    samples/fs_hz/window.

    The transform length equals the actual sample count. Amplitude
    scaling uses the window sum; it corrects coherent gain, not general
    leakage bias.

    Args:
        samples: Finite real time samples to transform.
        fs_hz: Positive sampling frequency in Hz.
        window: Nonnegative window weights with the same shape as
            samples.

    Returns:
        Hz bin centres, unscaled real DFT coefficients, and gain-
        corrected single-sided amplitudes.
    """
    if samples.ndim != 1 or window.shape != samples.shape:
        raise ValueError("Samples and window must be equal-length 1D arrays.")
    if np.iscomplexobj(samples) or not np.all(np.isfinite(samples)):
        raise ValueError("Samples must be real and finite.")
    if fs_hz <= 0 or not np.isfinite(fs_hz) or samples.size < 3:
        raise ValueError(
            "A positive sample rate and at least three points are needed."
        )
    if (
        not np.all(np.isfinite(window))
        or np.any(window < 0)
        or window.sum() <= 0
    ):
        raise ValueError(
            "The analysis window must be finite, nonnegative, nonzero."
        )
    count = samples.size
    transformed = np.fft.rfft(samples * window, norm="backward")
    amplitude = np.abs(transformed) / window.sum()
    if count % 2 == 0:
        amplitude[1:-1] *= 2.0
    else:
        amplitude[1:] *= 2.0
    frequencies = np.fft.rfftfreq(count, d=1.0 / fs_hz)
    return frequencies, transformed, amplitude


def db_relative(amplitude: FloatArray, reference: float) -> FloatArray:
    """Return 20 log10(amplitude/reference); protect exact zero only
    numerically.

    Args:
        amplitude: Nonnegative single-sided amplitudes.
        reference: Positive common amplitude reference defining 0 dB.

    Returns:
        Amplitude ratios in decibels; exact zero is protected
        numerically.
    """
    if reference <= 0:
        raise ValueError("The dB reference must be positive.")
    return 20 * np.log10(
        np.maximum(amplitude / reference, np.finfo(float).tiny)
    )


def component_peaks(
    amplitude: FloatArray, reference: float, threshold_db: float
) -> IndexArray:
    """Return local-maximum indices above threshold_db relative to
    reference.

    The two endpoints are tested separately. The left-strict, right-
    nonstrict rule selects one side of a possible flat local maximum.
    This is an operational detector for this clean, resolved multitone
    record.

    Args:
        amplitude: Single-sided amplitudes on the native FFT grid.
        reference: Positive common amplitude reference for thresholding.
        threshold_db: Local-peak threshold relative to reference in
            decibels.

    Returns:
        Indices of local maxima passing the declared amplitude
        threshold.
    """
    peaks = np.zeros(amplitude.size, dtype=bool)
    peaks[1:-1] = (amplitude[1:-1] > amplitude[:-2]) & (
        amplitude[1:-1] >= amplitude[2:]
    )
    peaks[0] = amplitude[0] > amplitude[1]
    peaks[-1] = amplitude[-1] > amplitude[-2]
    peaks &= amplitude >= reference * 10 ** (threshold_db / 20)
    return np.flatnonzero(peaks)


def one_sided_power(transformed: ComplexArray, count: int) -> FloatArray:
    """Return rectangular-window mean-square contributions from raw
    DFT/count.

    Args:
        transformed: Raw rectangular-window rFFT coefficients.
        count: Number of original time-domain samples.

    Returns:
        One-sided mean-square contributions with conjugate bins paired.
    """
    power = np.abs(transformed / count) ** 2
    if count % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    return power


def numerical_checks(
    samples: FloatArray,
    transformed: ComplexArray,
    power_bins: FloatArray,
    peaks: IndexArray,
) -> list[dict]:
    """Return formula-based calibration and cross-check results for the
    measured data.

    Args:
        samples: Original reference amplitudes for the verification.
        transformed: Raw rFFT coefficients measured from samples.
        power_bins: Rectangular-window one-sided power contributions.
        peaks: Detected component indices for selected explicit DFT
            checks.

    Returns:
        Result dictionaries containing errors, tolerances, units and
        flags.
    """
    rows: list[dict] = []

    def check(name: str, error: float, tolerance: float, unit: str) -> None:
        """Record numerical error against tolerance in unit, and fail if
        exceeded.

        Args:
            name: Description of the tested identity.
            error: Nonnegative numerical discrepancy.
            tolerance: Largest accepted discrepancy in the stated unit.
            unit: Unit or normalisation used for the discrepancy.

        Returns:
            None. Record the result and raise if the identity fails.
        """
        passed = bool(np.isfinite(error) and error <= tolerance)
        rows.append(
            {
                "check": name,
                "error": error,
                "tolerance": tolerance,
                "unit": unit,
                "passed": passed,
            }
        )
        if not passed:
            raise ArithmeticError(name)

    # Analytic inputs follow DFT example; these are separate from the reference.
    for name, count, k, amplitude, phase in [
        ("DC is not doubled", 32, 0, 3.0, 0.0),
        ("even Nyquist is not doubled", 32, 16, 2.5, 0.0),
        ("interior real cosine recovers amplitude", 64, 7, 2.0, 0.7),
        ("odd final bin is doubled", 63, 31, 1.2, 0.4),
    ]:
        n = np.arange(count)
        calibration = amplitude * np.cos(2 * np.pi * k * n / count + phase)
        _, _, measured = spectrum(calibration, float(count), np.ones(count))
        check(name, abs(float(measured[k]) - amplitude), 1e-12, "amplitude")
    count = samples.size
    raw_full = np.fft.fft(samples, norm="backward")
    scale = max(1.0, float(np.max(np.abs(transformed))))
    check(
        "rFFT agrees with full FFT nonnegative bins",
        float(np.max(np.abs(raw_full[: transformed.size] - transformed)))
        / scale,
        1e-12,
        "relative coefficient",
    )
    # A few actual long-record bins are independently evaluated by the DFT sum.
    selected = np.unique([0, int(peaks[0]), int(peaks[-1]), count // 2])
    n = np.arange(count)
    direct = np.array(
        [
            np.sum(samples * np.exp(-2j * np.pi * k * n / count))
            for k in selected
        ]
    )
    check(
        "selected bins agree with explicit reference DFT sum",
        float(np.max(np.abs(direct - transformed[selected]))) / scale,
        1e-10,
        "relative coefficient",
    )
    recovered = np.fft.irfft(transformed, n=count, norm="backward")
    check(
        "same-grid FFT inverse round trip",
        float(np.max(np.abs(recovered - samples))),
        1e-12,
        "amplitude",
    )
    time_power = float(np.mean(samples**2))
    check(
        "Parseval mean-square consistency",
        abs(float(power_bins.sum()) - time_power) / time_power,
        1e-12,
        "relative power",
    )
    n = np.arange(count)
    explicit_hann = 0.5 * (1 - np.cos(2 * np.pi * n / (count - 1)))
    check(
        "NumPy Hann matches reference L-1 formula",
        float(np.max(np.abs(np.hanning(count) - explicit_hann))),
        1e-14,
        "window weight",
    )
    check(
        "symmetric Hann coherent gain",
        abs(float(np.hanning(count).mean()) - (count - 1) / (2 * count)),
        1e-14,
        "gain",
    )
    return rows


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save fig in figures/name as 300 dpi PNG and vector SVG; return None.

    Args:
        fig: Matplotlib figure to export.
        name: Filename stem beneath the figures directory.

    Returns:
        None. Save 300 dpi PNG and SVG files, then close the figure.
    """
    fig.savefig(ROOT / "figures" / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_results(
    samples: FloatArray,
    frequencies: FloatArray,
    rectangular: FloatArray,
    hann: FloatArray,
    peaks: IndexArray,
    reference: float,
) -> None:
    """Save the measured waveform, spectrum and matched-scale window
    comparison.

    Args:
        samples: Original reference samples for the waveform excerpt.
        frequencies: Native one-sided FFT bin centres in Hz.
        rectangular: Measured rectangular-window amplitudes.
        hann: Coherent-gain-corrected symmetric Hann amplitudes.
        peaks: Detected source-component indices.
        reference: Common amplitude defining the relative dB scale.

    Returns:
        None. Save waveform, spectrum and window-comparison figures.
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    count = samples.size
    band_edge = float(frequencies[peaks[-1]])
    fig, axes = plt.subplots(
        2, 1, figsize=(9, 6.2), layout="constrained", sharex=True
    )
    axes[0].plot(frequencies / 1000, rectangular, color="#225d91", lw=0.8)
    axes[0].scatter(
        frequencies[peaks] / 1000,
        rectangular[peaks],
        s=20,
        color="#225d91",
        zorder=4,
    )
    for k in peaks:
        axes[0].annotate(
            f"{frequencies[k]:g} Hz\nA={rectangular[k]:.3g}",
            (frequencies[k] / 1000, rectangular[k]),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=9,
        )
    axes[0].set(
        ylabel="Single-sided amplitude (NPY units)",
        ylim=(-0.03, 1.2 * reference),
        title=(
            f"Measured reference spectrum | L={count:,}, "
            f"$F_0$={FS_HZ:g} Hz, rectangular"
        ),
    )
    display_floor = -140.0
    spectrum_db = db_relative(rectangular, reference)
    axes[1].plot(
        frequencies / 1000,
        np.maximum(spectrum_db, display_floor),
        color="#225d91",
        lw=0.8,
    )
    axes[1].scatter(
        frequencies[peaks] / 1000,
        spectrum_db[peaks],
        s=20,
        color="#225d91",
        zorder=4,
    )
    axes[1].axhline(
        THRESHOLD_DB,
        ls="--",
        color="#a65a1d",
        lw=1,
        label="Detection threshold: -60 dB",
    )
    axes[1].set(
        xlabel="Frequency (kHz)",
        ylabel="Amplitude (dB relative to largest peak)",
        ylim=(display_floor - 4, 8),
        xlim=(0, FS_HZ / 2000),
    )
    axes[1].text(
        0.98,
        0.70,
        "Display floor: -140 dB (not measured noise)\n"
        "Native grid: 0.5 Hz; no zero-padding",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    axes[1].legend(loc="upper right", fontsize=9)
    for ax in axes:
        ax.axvline(band_edge / 1000, color="#9b3544", ls=":", lw=1)
        ax.grid(alpha=0.2)
    save_figure(fig, "reference_spectrum")
    fig, axes = plt.subplots(
        1, 2, figsize=(9, 3.9), layout="constrained", sharey=True
    )
    for ax, k in zip(axes, [peaks[0], peaks[-1]]):
        selected = np.abs(frequencies - frequencies[k]) <= 2.0
        offsets = frequencies[selected] - frequencies[k]
        ax.plot(
            offsets,
            rectangular[selected] / rectangular[k],
            "o-",
            ms=4,
            lw=1,
            color="#225d91",
            label="Rectangular",
        )
        ax.plot(
            offsets,
            hann[selected] / rectangular[k],
            "s--",
            ms=4,
            lw=1,
            color="#d47b24",
            label="Symmetric Hann, gain corrected",
        )
        ax.set(
            title=f"Around the {frequencies[k]:g} Hz component",
            xlabel="Offset from component centre (Hz)",
            xticks=np.arange(-2, 2.1, 0.5),
        )
        ax.grid(alpha=0.2)
    axes[0].set(
        ylabel="Amplitude / rectangular component amplitude", ylim=(-0.03, 1.1)
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        fontsize=9,
        loc="outside lower center",
        ncol=2,
    )
    fig.suptitle(
        "Same 2 s record and native 0.5 Hz bins; "
        "lines only connect measured bins",
        fontsize=10,
    )
    save_figure(fig, "window_comparison")
    fig, ax = plt.subplots(figsize=(9, 2.7), layout="constrained")
    preview = min(count, int(0.02 * FS_HZ))
    ax.plot(
        1000 * np.arange(preview) / FS_HZ,
        samples[:preview],
        lw=0.9,
        color="#225d91",
    )
    ax.set(
        xlabel="Time (ms)",
        ylabel="Amplitude (NPY units)",
        title="Original reference samples: first 20 ms (display excerpt only)",
    )
    ax.grid(alpha=0.2)
    save_figure(fig, "reference_waveform")


def main() -> None:
    """Read the fixed reference and write all R1.1 measurements and
    figures.

    Returns:
        None. Write the described outputs or raise on invalid data.
    """
    for name in ["results", "figures"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    input_path = PROJECT / "reference_signal.npy"
    before_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    original = np.load(input_path, allow_pickle=False)
    if original.ndim != 1 or np.iscomplexobj(original) or original.size < 3:
        raise ValueError("A one-dimensional real reference is required.")
    samples = np.asarray(original, dtype=np.float64)
    if not np.all(np.isfinite(samples)) or not np.any(samples):
        raise ValueError("The reference must be finite and nonzero.")
    with wave.open(str(PROJECT / "reference_signal.wav"), "rb") as wav:
        wav_header = {
            "sample_rate_hz": wav.getframerate(),
            "channels": wav.getnchannels(),
            "frames": wav.getnframes(),
            "sample_width_bytes": wav.getsampwidth(),
        }
    if (
        wav_header["sample_rate_hz"] != FS_HZ
        or wav_header["frames"] != samples.size
        or wav_header["channels"] != 1
    ):
        raise ValueError(
            "NPY length/reference sample rate and WAV metadata disagree."
        )
    count = samples.size
    rectangular_window = np.ones(count)
    hann_window = np.hanning(count)
    frequencies, transformed, rectangular = spectrum(
        samples, FS_HZ, rectangular_window
    )
    _, _, hann = spectrum(samples, FS_HZ, hann_window)
    reference = float(rectangular.max())
    rectangular_db = db_relative(rectangular, reference)
    peaks = component_peaks(rectangular, reference, THRESHOLD_DB)
    if peaks.size == 0:
        raise ArithmeticError("No component passed the stated detection rule.")
    power_bins = one_sided_power(transformed, count)
    time_power = float(np.mean(samples**2))
    checks = numerical_checks(samples, transformed, power_bins, peaks)
    save_csv("checks.csv", checks)
    components = []
    t = np.arange(count) / FS_HZ
    fitted = np.zeros(count)
    for k in peaks:
        phase = float(np.angle(transformed[k]))
        fitted += rectangular[k] * np.cos(
            2 * np.pi * frequencies[k] * t + phase
        )
        components.append(
            {
                "bin": int(k),
                "frequency_hz": float(frequencies[k]),
                "amplitude_npy_units": float(rectangular[k]),
                "phase_cosine_rad": phase,
                "relative_db": float(rectangular_db[k]),
                "cycles_in_record": float(frequencies[k] * count / FS_HZ),
                "mean_square_contribution": float(power_bins[k]),
                "power_fraction": float(power_bins[k] / time_power),
                "hann_amplitude_gain_corrected": float(hann[k]),
                "hann_minus_rectangular_amplitude": float(
                    hann[k] - rectangular[k]
                ),
            }
        )
    save_csv("components.csv", components)
    sensitivity = []
    for threshold in SENSITIVITY_DB:
        for name, amplitude in [
            ("rectangular", rectangular),
            ("symmetric_hann", hann),
        ]:
            detected = component_peaks(amplitude, reference, threshold)
            above = np.flatnonzero(
                amplitude >= reference * 10 ** (threshold / 20)
            )
            sensitivity.append(
                {
                    "window": name,
                    "threshold_db": threshold,
                    "peak_count": detected.size,
                    "peak_frequencies_hz": ";".join(
                        f"{frequencies[k]:g}" for k in detected
                    ),
                    "highest_peak_centre_hz": float(frequencies[detected[-1]]),
                    "all_bins_above_threshold": above.size,
                    "last_bin_above_threshold_hz": float(
                        frequencies[above[-1]]
                    ),
                }
            )
    save_csv("bandwidth_sensitivity.csv", sensitivity)
    # Record a secondary bandwidth definition to explain the primary choice.
    cumulative = np.cumsum(power_bins) / power_bins.sum()
    save_csv(
        "alternative_energy_bandwidth.csv",
        [
            {
                "power_fraction_target": q,
                "first_frequency_reaching_target_hz": float(
                    frequencies[np.searchsorted(cumulative, q)]
                ),
            }
            for q in [0.99, 0.999]
        ],
    )
    save_csv(
        "spectrum.csv",
        [
            {
                "frequency_hz": float(f),
                "rectangular_amplitude": float(a),
                "hann_amplitude_gain_corrected": float(h),
                "rectangular_relative_db": float(d),
                "rectangular_power_contribution": float(p),
            }
            for f, a, h, d, p in zip(
                frequencies, rectangular, hann, rectangular_db, power_bins
            )
        ],
    )
    save_csv(
        "waveform_excerpt.csv",
        [
            {
                "sample_index": n,
                "time_s": n / FS_HZ,
                "amplitude": float(samples[n]),
            }
            for n in range(min(count, int(0.02 * FS_HZ)))
        ],
    )
    outside = np.ones(rectangular.size, dtype=bool)
    outside[peaks] = False
    residual = samples - fitted
    metrics = {
        "sample_rate_hz": FS_HZ,
        "sample_count": count,
        "record_duration_s": count / FS_HZ,
        "last_timestamp_s": (count - 1) / FS_HZ,
        "fft_length": count,
        "bin_spacing_hz": FS_HZ / count,
        "amplitude_minimum": float(samples.min()),
        "amplitude_maximum": float(samples.max()),
        "mean": float(samples.mean()),
        "mean_square": time_power,
        "rms": float(np.sqrt(time_power)),
        "threshold_db": THRESHOLD_DB,
        "detected_components": peaks.size,
        "band_edge_hz": float(frequencies[peaks[-1]]),
        "hann_coherent_gain": float(hann_window.mean()),
        "max_hann_peak_amplitude_difference": float(
            np.max(np.abs(hann[peaks] - rectangular[peaks]))
        ),
        "maximum_other_bin_db": float(rectangular_db[outside].max()),
        "other_bins_power_fraction": float(
            power_bins[outside].sum() / power_bins.sum()
        ),
        "measured_tone_model_residual_rms": float(
            np.sqrt(np.mean(residual**2))
        ),
        "measured_tone_model_residual_maximum": float(
            np.max(np.abs(residual))
        ),
        "deterministic_check_count": len(checks),
    }
    save_csv(
        "summary.csv", [{"metric": k, "value": v} for k, v in metrics.items()]
    )
    metadata = {
        "input_npy_sha256": before_hash,
        "input_original_dtype": str(original.dtype),
        "sample_rate_source": (
            "Externally provided reference signal, cross-checked with WAV header"
        ),
        "wav_header": wav_header,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "seed": "None; deterministic analysis",
        "primary_window": "rectangular",
        "comparison_window": "symmetric Hann, denominator L-1",
        "zero_padding": False,
        "bandwidth_definition": (
            "highest resolved component peak at/above -60 dB "
            "relative to largest rectangular peak"
        ),
        "sensitivity_thresholds_db": list(SENSITIVITY_DB),
        "display_floor_db": -140,
        "limitations": (
            "Operational component band edge of the supplied finite clean "
            "record; not a claim of strict compact support of a "
            "time-truncated DTFT or physical analog signal."
        ),
    }
    (ROOT / "results" / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_results(samples, frequencies, rectangular, hann, peaks, reference)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != before_hash:
        raise ArithmeticError("Reference input changed during analysis.")
    print(json.dumps(metrics, indent=2))
    print("Component frequencies (Hz):", frequencies[peaks].tolist())


if __name__ == "__main__":
    main()
