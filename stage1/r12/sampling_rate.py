"""Compare transmission rates using the sampling model and R1.1 band.

Run from the project root with ``python -m stage1.r12.sampling_rate``.
The clean sampling experiment uses no random degradation. Fourier interpolation
is an explicitly derived periodic-record verification, not a general-purpose resampling algorithm.
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
from scipy.signal import resample

from stage1.r11.characterise_reference import FS_HZ, THRESHOLD_DB, spectrum

matplotlib.use("Agg")

FloatArray = NDArray[np.float64]
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
FACTORS = (4, 3, 2, 1)


def save_csv(name: str, rows: list[dict]) -> None:
    """Write rows with their field names to results/name; return None.

    Args:
        name: Output filename beneath the R1.2 results directory.
        rows: Nonempty result dictionaries sharing the same field names.

    Returns:
        None. Write the measurement header and rows.
    """
    with (ROOT / "results" / name).open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def periodic_reconstruct(samples: FloatArray, target_count: int) -> FloatArray:
    """Interpolate real, even-length samples to target_count points in one
    period.

    target_count must be even and at least the input length. The output
    uses only samples; the original reference and measured source tones
    are not inputs.

    Args:
        samples: Finite real even-length samples for one declared
            period.
        target_count: Even output length, at least the input sample
            count.

    Returns:
        Fourier-interpolated amplitudes over the same period.
    """
    values = np.asarray(samples)
    if (
        values.ndim != 1
        or values.size < 4
        or values.size % 2
        or np.iscomplexobj(values)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "Real finite even-length samples (at least four) are needed."
        )
    if (
        not isinstance(target_count, (int, np.integer))
        or target_count < values.size
        or target_count % 2
    ):
        raise ValueError(
            "target_count must be an even integer >= input length."
        )
    count = values.size
    transformed = np.fft.rfft(values, norm="backward")
    padded = np.zeros(target_count // 2 + 1, dtype=np.complex128)
    padded[: transformed.size] = transformed * (target_count / count)
    if target_count > count:
        # The old self-conjugate endpoint becomes an interior conjugate pair.
        padded[count // 2] *= 0.5
    return np.fft.irfft(padded, n=target_count, norm="backward")


def mathematical_checks() -> list[dict]:
    """Return independent DC, phase, Nyquist, grid and SciPy interpolation
    checks.

    Returns:
        Independent mathematical check result dictionaries.
    """
    rows = []
    count, target = 32, 96
    for name, amplitude, k, phase in [
        ("DC amplitude", 2.3, 0, 0.0),
        ("interior cosine amplitude and phase", 0.7, 5, 0.43),
        ("input Nyquist coefficient split", 0.4, count // 2, 0.0),
    ]:
        samples = amplitude * np.cos(
            2 * np.pi * k * np.arange(count) / count + phase
        )
        expected = amplitude * np.cos(
            2 * np.pi * k * np.arange(target) / target + phase
        )
        recovered = periodic_reconstruct(samples, target)
        rows.append(
            check_row(name, float(np.max(np.abs(recovered - expected))), 2e-14)
        )
    samples = 0.2 + np.cos(2 * np.pi * 5 * np.arange(count) / count + 0.2)
    samples += 0.3 * (-1.0) ** np.arange(count)
    rows.append(
        check_row(
            "same-grid reconstruction",
            float(
                np.max(np.abs(periodic_reconstruct(samples, count) - samples))
            ),
            2e-14,
        )
    )
    rows.append(
        check_row(
            "independent SciPy Fourier resampling cross-check",
            float(
                np.max(
                    np.abs(
                        periodic_reconstruct(samples, target)
                        - resample(samples, target)
                    )
                )
            ),
            2e-14,
        )
    )
    return rows


def check_row(name: str, error: float, tolerance: float) -> dict:
    """Return a name/error/tolerance/pass row; raise if error exceeds
    tolerance.

    Args:
        name: Description of the mathematical check.
        error: Nonnegative observed discrepancy.
        tolerance: Largest accepted discrepancy for this check.

    Returns:
        A name/error/tolerance/pass dictionary; raise if it fails.
    """
    passed = bool(np.isfinite(error) and error <= tolerance)
    if not passed:
        raise ArithmeticError(f"{name}: {error:g} exceeds {tolerance:g}")
    return {
        "check": name,
        "error": error,
        "tolerance": tolerance,
        "passed": passed,
    }


def save_figure(fig: plt.Figure, name: str) -> None:
    """Write fig to figures/name as 300 dpi PNG and SVG; return None.

    Args:
        fig: Matplotlib figure to export.
        name: Output filename stem beneath the figures directory.

    Returns:
        None. Save 300 dpi PNG and SVG files, then close the figure.
    """
    fig.savefig(ROOT / "figures" / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_evidence(
    reference: FloatArray,
    selected_samples: FloatArray,
    reconstructed: FloatArray,
    selected_fs: float,
    band_edge: float,
    rates: list[dict],
) -> None:
    """Plot spectral retention, sample cost and reconstruction; return None.

    Args:
        reference: Original 48 kHz reference amplitudes.
        selected_samples: Samples obtained at selected_fs.
        reconstructed: Reconstructed values on the original reference grid.
        selected_fs: Selected transmission sampling rate in Hz.
        band_edge: Measured highest retained component frequency in Hz.
        rates: Candidate-rate measurements, flags and relative payloads.

    Returns:
        None. Two PNG/SVG figure pairs are written to the figures directory.
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    f_ref, _, a_ref = spectrum(reference, FS_HZ, np.ones(reference.size))
    f_low, _, a_low = spectrum(
        selected_samples, selected_fs, np.ones(selected_samples.size)
    )
    display_threshold = np.max(a_ref) * 10 ** (THRESHOLD_DB / 20)
    mask = a_ref >= display_threshold
    selected_mask = a_low >= display_threshold
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), layout="constrained")
    axes[0].vlines(
        f_ref[mask] / 1000,
        0,
        a_ref[mask],
        color="#9aa0a6",
        lw=5,
        label="48 kHz reference",
    )
    axes[0].scatter(
        f_low[selected_mask] / 1000,
        a_low[selected_mask],
        color="#225d91",
        s=30,
        zorder=3,
        label=f"{selected_fs/1000:g} kHz samples",
    )
    axes[0].axvline(
        band_edge / 1000, color="#a65a1d", ls="--", label="Measured band edge"
    )
    axes[0].set(
        xlim=(0, selected_fs / 2000),
        ylim=(0, 1.12),
        xlabel="Frequency (kHz)",
        ylabel="Single-sided amplitude (NPY units)",
        title="All four source components retained",
    )
    axes[0].legend(fontsize=8, loc="upper right")
    labels = [f"{r['sample_rate_hz']/1000:g}" for r in rates]
    colors = [
        (
            "#b45b4c"
            if not r["strict_nyquist"]
            else "#225d91" if r["selected"] else "#a0acb8"
        )
        for r in rates
    ]
    bars = axes[1].bar(
        labels, [r["sample_count"] / 1000 for r in rates], color=colors
    )
    for bar, row in zip(bars, rates):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{row['payload_ratio_to_selected']:g}x",
            ha="center",
            fontsize=9,
        )
    axes[1].text(
        0.04,
        0.94,
        "Red: fails strict Nyquist condition\n"
        "Blue: selected integer-decimation rate",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
    )
    axes[1].set(
        xlabel="Sampling rate (kHz)",
        ylabel="Samples in the same 2 s record (thousands)",
        ylim=(0, 115),
        title="Raw payload at fixed bits/sample",
    )
    for ax in axes:
        ax.grid(alpha=0.2, axis="y")
    save_figure(fig, "sampling_rate_decision")

    t = np.arange(reference.size) / FS_HZ
    low_t = np.arange(selected_samples.size) / selected_fs
    visible = t < 0.004
    low_visible = low_t < 0.004
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6), layout="constrained")
    axes[0].plot(
        t[visible] * 1000,
        reference[visible],
        color="#a65a1d",
        lw=1.8,
        label="48 kHz reference",
    )
    axes[0].plot(
        t[visible] * 1000,
        reconstructed[visible],
        color="#225d91",
        ls="--",
        lw=1.2,
        label="Periodic Fourier reconstruction",
    )
    axes[0].scatter(
        low_t[low_visible] * 1000,
        selected_samples[low_visible],
        color="#225d91",
        s=12,
        label=f"{selected_fs/1000:g} kHz input samples",
    )
    axes[0].set(
        xlabel="Time (ms)",
        ylabel="Amplitude (NPY units)",
        title="First 4 ms shown; entire 2 s evaluated",
    )
    axes[0].legend(fontsize=8, ncol=1, loc="lower left")
    axes[1].plot(t, reconstructed - reference, color="#225d91", lw=0.6)
    axes[1].set(
        xlabel="Time (s)",
        ylabel="Reconstruction error (NPY units)",
        title="Full-record error; no time shift, fitting or gain adjustment",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    save_figure(fig, "selected_rate_reconstruction")


def main() -> None:
    """Read reference/R1.1, write R1.2 measurements and plots, and return
    None.

    Returns:
        None. Write the described outputs or raise on invalid data.
    """
    for directory in ["results", "figures"]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    input_path = PROJECT / "reference_signal.npy"
    original_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    reference = np.load(input_path, allow_pickle=False)
    r11 = ROOT.parent / "r11" / "results"
    metadata = json.loads((r11 / "metadata.json").read_text(encoding="utf-8"))
    if metadata["input_npy_sha256"] != original_hash:
        raise ValueError(
            "R1.1 measurements refer to a different reference file."
        )
    with (r11 / "summary.csv").open(encoding="utf-8", newline="") as stream:
        measured = {
            r["metric"]: float(r["value"]) for r in csv.DictReader(stream)
        }
    with (r11 / "components.csv").open(encoding="utf-8", newline="") as stream:
        components = list(csv.DictReader(stream))
    band_edge = measured["band_edge_hz"]
    if (
        reference.ndim != 1
        or np.iscomplexobj(reference)
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError("A finite real mono reference is needed.")
    if (
        measured["sample_rate_hz"] != FS_HZ
        or measured["sample_count"] != reference.size
    ):
        raise ValueError("R1.1 sampling metadata disagree.")
    admissible = [m for m in FACTORS if FS_HZ / m > 2 * band_edge]
    selected_factor = max(admissible)
    selected_fs = FS_HZ / selected_factor
    duration = reference.size / FS_HZ
    reference_fft = np.fft.rfft(reference)
    checks = mathematical_checks()
    rates, component_rows, spectra = [], [], []
    reconstructions = {}
    for factor in FACTORS:
        if reference.size % factor:
            raise ValueError(
                "All compared records must have exactly the same duration."
            )
        fs_hz = FS_HZ / factor
        samples = reference[::factor].copy()
        count = samples.size
        frequencies, transformed, amplitudes = spectrum(
            samples, fs_hz, np.ones(count)
        )
        reconstructed = periodic_reconstruct(samples, reference.size)
        reconstructions[factor] = reconstructed
        error = reconstructed - reference
        mse = float(np.mean(error**2))
        rates.append(
            {
                "sample_rate_hz": fs_hz,
                "decimation_factor": factor,
                "sample_count": count,
                "duration_s": count / fs_hz,
                "nyquist_frequency_hz": fs_hz / 2,
                "guard_to_nyquist_hz": fs_hz / 2 - band_edge,
                "replica_gap_hz": fs_hz - 2 * band_edge,
                "strict_nyquist": bool(fs_hz > 2 * band_edge),
                "selected": factor == selected_factor,
                "payload_ratio_to_selected": fs_hz / selected_fs,
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "nmse": mse / float(np.mean(reference**2)),
                "maximum_absolute_error": float(np.max(np.abs(error))),
                "interpolation_at_original_sample_error": float(
                    np.max(np.abs(reconstructed[::factor] - samples))
                ),
            }
        )
        for comp in components:
            frequency = float(comp["frequency_hz"])
            k = int(round(frequency * duration))
            phase = float(comp["phase_cosine_rad"])
            amplitude = float(comp["amplitude_npy_units"])
            component_rows.append(
                {
                    "sample_rate_hz": fs_hz,
                    "source_frequency_hz": frequency,
                    "measured_bin": k,
                    "measured_frequency_hz": float(frequencies[k]),
                    "measured_amplitude": float(amplitudes[k]),
                    "source_amplitude": amplitude,
                    "endpoint": bool(k == count // 2),
                    "normalised_coefficient_real": float(
                        transformed[k].real / count
                    ),
                    "normalised_coefficient_imag": float(
                        transformed[k].imag / count
                    ),
                    "cosine_phase_rad": (
                        float(np.angle(transformed[k]))
                        if k < count // 2
                        else "Nyquist endpoint; no independent sine phase"
                    ),
                    "nyquist_cosine_prediction": (
                        amplitude * np.cos(phase)
                        if k == count // 2
                        else "not applicable"
                    ),
                    "normalised_coefficient_absolute_error": (
                        float(
                            abs(
                                transformed[k] / count
                                - reference_fft[k] / reference.size
                            )
                        )
                        if k < count // 2
                        else "not comparable as conjugate pair"
                    ),
                }
            )
        spectra.extend(
            {
                "sample_rate_hz": fs_hz,
                "frequency_hz": float(f),
                "single_sided_amplitude": float(a),
            }
            for f, a in zip(frequencies, amplitudes)
        )
    selected_samples = reference[::selected_factor].copy()
    selected_reconstruction = reconstructions[selected_factor]
    selected = next(row for row in rates if row["selected"])
    checks.extend(
        [
            check_row(
                "selected low-rate sample coordinates match reference stride",
                float(
                    np.max(
                        np.abs(
                            np.arange(selected_samples.size) / selected_fs
                            - np.arange(0, reference.size, selected_factor)
                            / FS_HZ
                        )
                    )
                ),
                5e-16,
            ),
            check_row(
                "selected full-record duration",
                abs(selected_samples.size / selected_fs - duration),
                1e-15,
            ),
            check_row(
                "selected component complex coefficients retained",
                max(
                    r["normalised_coefficient_absolute_error"]
                    for r in component_rows
                    if r["sample_rate_hz"] == selected_fs
                ),
                1e-12,
            ),
            check_row(
                "selected reconstruction maximum error",
                selected["maximum_absolute_error"],
                1e-10,
            ),
            check_row(
                "selected reconstruction at its input samples",
                selected["interpolation_at_original_sample_error"],
                1e-12,
            ),
        ]
    )
    critical_fs = 2 * band_edge
    critical = next(
        (r for r in rates if r["sample_rate_hz"] == critical_fs), None
    )
    if critical is not None:
        edge = next(
            c
            for c in component_rows
            if c["sample_rate_hz"] == critical_fs
            and c["source_frequency_hz"] == band_edge
        )
        expected_missing_power = edge["source_amplitude"] ** 2 / 2
        checks.append(
            check_row(
                "critical Nyquist endpoint agrees with "
                "cosine-phase prediction",
                abs(
                    edge["normalised_coefficient_real"]
                    - edge["nyquist_cosine_prediction"]
                ),
                1e-11,
            )
        )
        checks.append(
            check_row(
                "critical-rate error equals missing highest-tone power",
                abs(critical["mse"] - expected_missing_power),
                1e-12,
            )
        )
    summary = {
        "reference_sample_rate_hz": FS_HZ,
        "reference_sample_count": reference.size,
        "duration_s": duration,
        "measured_band_edge_hz": band_edge,
        "nyquist_rate_hz": 2 * band_edge,
        "selected_sample_rate_hz": selected_fs,
        "selected_factor": selected_factor,
        "selected_sample_count": selected_samples.size,
        "selected_nyquist_frequency_hz": selected_fs / 2,
        "selected_guard_hz": selected_fs / 2 - band_edge,
        "selected_replica_gap_hz": selected_fs - 2 * band_edge,
        "selected_mse": selected["mse"],
        "selected_nmse": selected["nmse"],
        "selected_maximum_absolute_error": selected["maximum_absolute_error"],
        "payload_reduction_vs_reference_percent": 100
        * (1 - selected_fs / FS_HZ),
        "deterministic_check_count": len(checks),
    }
    save_csv("rate_comparison.csv", rates)
    save_csv("component_retention.csv", component_rows)
    save_csv("candidate_spectra.csv", spectra)
    save_csv("checks.csv", checks)
    save_csv(
        "summary.csv", [{"metric": k, "value": v} for k, v in summary.items()]
    )
    save_csv(
        "selected_samples.csv",
        [
            {"index": m, "time_s": m / selected_fs, "value": float(v)}
            for m, v in enumerate(selected_samples)
        ],
    )
    save_csv(
        "selected_reconstruction.csv",
        [
            {
                "index": n,
                "time_s": n / FS_HZ,
                "reference": float(v),
                "reconstructed": float(selected_reconstruction[n]),
                "error": float(selected_reconstruction[n] - v),
            }
            for n, v in enumerate(reference)
        ],
    )
    np.save(
        ROOT / "results" / "selected_samples.npy",
        selected_samples,
        allow_pickle=False,
    )
    metadata = {
        "input_npy_sha256": original_hash,
        "rate_selection": (
            "Lowest strictly sufficient rate among reference[::M] "
            "candidates; integer-only family is our design choice, "
            "not a physical constraint or global optimum."
        ),
        "decimation_factors": list(FACTORS),
        "selected_factor": selected_factor,
        "pre_decimation_filter": (
            "None for this measured clean reference; sub-Nyquist source "
            "content is verified. Not a general unfiltered-decimation rule."
        ),
        "reconstruction": (
            "Real periodic Fourier interpolation, same 2 s period; "
            "scale L/N and split old Nyquist coefficient when expanded."
        ),
        "evaluation": (
            "All 96000 original grid positions, no trimming, fitted gain, "
            "phase adjustment or time alignment."
        ),
        "random_seed": None,
        "limitations": (
            "Periodic-record clean-sampling baseline, not a claim of "
            "perfect reconstruction for nonperiodic/noisy data or a "
            "practical causal filter. Full stochastic system evaluation "
            "remains pending."
        ),
    }
    (ROOT / "results" / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_evidence(
        reference,
        selected_samples,
        selected_reconstruction,
        selected_fs,
        band_edge,
        rates,
    )
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != original_hash:
        raise ArithmeticError("The original reference changed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
