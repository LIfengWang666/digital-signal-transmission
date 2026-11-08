"""Measure R2.1 method-selection evidence using the linear convolution model.

Prototype parameters support the approach decision, not a completed R2.2
optimisation or held-out R2.3 demonstration. All noise uses design phase 0.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from degradation import add_gaussian_noise  # noqa: E402

FloatArray = NDArray[np.float64]


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write stable, header-based measurement rows.

    Args:
        path: Destination CSV.
        rows: Nonempty records with identical ordered keys.

    Returns:
        None. Save the records.
    """
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_rng(path: tuple[int, ...]) -> np.random.Generator:
    """Create the recorded anchor-seed leaf from the accepted registry.

    Args:
        path: Full purpose/study/phase/condition/repeat spawn key.

    Returns:
        A fresh PCG64 Generator at the beginning of this leaf.
    """
    registry = json.loads(
        (PACKAGE / "metadata/randomness_design.json").read_text(
            encoding="utf-8"
        )
    )
    seed = np.random.SeedSequence(
        registry["root_seed"],
        spawn_key=path,
        pool_size=registry["pool_size"],
    )
    return np.random.Generator(np.random.PCG64(seed))


def linear_fft_convolution(
    samples: FloatArray, kernel: FloatArray
) -> FloatArray:
    """Compute full real linear convolution with sufficient zero padding.

    Args:
        samples: Real input record, zero outside its stored interval.
        kernel: Real finite impulse response, indexed from zero.

    Returns:
        N + M - 1 real outputs; raise on a material imaginary residual.
    """
    full_length = len(samples) + len(kernel) - 1
    fft_length = 1 << (full_length - 1).bit_length()
    spectrum = np.fft.fft(samples, fft_length)
    response = np.fft.fft(kernel, fft_length)
    result = np.fft.ifft(spectrum * response)
    if np.max(np.abs(result.imag)) > 1e-11:
        raise ValueError("Real convolution has a material imaginary part.")
    return result.real[:full_length]


def centred_filter(samples: FloatArray, kernel: FloatArray) -> FloatArray:
    """Apply an odd symmetric FIR with a predetermined centred output crop.

    Args:
        samples: Input record of N samples.
        kernel: Odd-length symmetric FIR with origin at its middle.

    Returns:
        N aligned samples from full[d:d + N], d = (M - 1) // 2.
    """
    if len(kernel) % 2 != 1 or not np.allclose(
        kernel, kernel[::-1], atol=1e-13, rtol=0
    ):
        raise ValueError(
            "The centred interface requires an odd symmetric FIR."
        )
    delay = (len(kernel) - 1) // 2
    full = linear_fft_convolution(samples, kernel)
    end = delay + len(samples)
    return full[delay:end]


def prototype_kernel(template: FloatArray, length: int) -> FloatArray:
    """Derive an illustrative finite FIR from an even frequency template.

    Args:
        template: Real even full-DFT gains on the actual record's grid.
        length: Odd number of centred impulse-response samples to retain.

    Returns:
        A real symmetric FIR tapered by the reference symmetric Hann.
    """
    if length % 2 != 1 or length > len(template):
        raise ValueError("Use an odd prototype length within the design grid.")
    if not np.array_equal(
        template, template[(-np.arange(len(template))) % len(template)]
    ):
        raise ValueError("The design template must be even.")
    impulse = np.fft.ifft(template)
    if np.max(np.abs(impulse.imag)) > 1e-13:
        raise ValueError(
            "The symmetric template did not produce a real kernel."
        )
    delay = (length - 1) // 2
    lags = np.arange(-delay, delay + 1)
    return impulse.real[lags % len(template)] * np.hanning(length)


def make_figures(results: Path, figures: Path) -> None:
    """Draw the argument from saved spectra, responses and design metrics.

    Args:
        results: Directory containing this run's CSV measurements.
        figures: Destination for PNG and SVG versions.

    Returns:
        None. Save two figures without recomputing their measurements.
    """

    def read(name: str) -> list[dict[str, str]]:
        """Read one saved CSV.

        Args:
            name: CSV filename relative to results.

        Returns:
            Header-keyed string records.
        """
        with (results / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    plt.rcdefaults()
    plt.style.use("seaborn-v0_8-whitegrid")
    spectra = read("pilot_spectrum.csv")
    responses = read("prototype_response.csv")
    noise = read("noise_trials.csv")
    anchor = next(
        r
        for r in noise
        if int(r["condition_id"]) == 5 and int(r["repeat_id"]) == 0
    )
    f = np.array([float(r["frequency_hz"]) for r in spectra])
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for key, label, color in [
        ("clean_amplitude", "Clean reference", "black"),
        ("noisy_amplitude", "Contaminated input", "tab:orange"),
    ]:
        values = np.array([float(r[key]) for r in spectra])
        axs[0].plot(
            f,
            20 * np.log10(np.maximum(values, 1e-7)),
            label=label,
            color=color,
            linewidth=0.8,
        )
    axs[0].set(
        ylabel="Single-sided amplitude (dB re 1 NPY unit)",
        ylim=(-80, 5),
        title=(
            "R2.1 design record: target SNR "
            f"{float(anchor['target_snr_db']):.3f} dB; realised "
            f"{float(anchor['realised_snr_db']):.3f} dB"
        ),
    )
    for method, label in [
        ("ma3", "Three-point mean"),
        ("wide_lowpass", "Wide low-pass prototype"),
        ("multiband", "Multiband prototype"),
    ]:
        rows = [r for r in responses if r["method"] == method]
        axs[1].plot(
            [float(r["frequency_hz"]) for r in rows],
            20
            * np.log10(
                np.maximum([float(r["magnitude"]) for r in rows], 1e-5)
            ),
            label=label,
            linewidth=1,
        )
    axs[1].set(
        xlabel="Frequency (Hz)",
        ylabel="Filter magnitude (dB re unity)",
        xlim=(0, 8000),
        ylim=(-80, 5),
        title=(
            "Realised FIR responses; "
            "prototype parameters remain provisional"
        ),
    )
    for ax in axs:
        ax.axvline(6000, color="grey", linestyle=":", linewidth=0.9)
        ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "approach_spectrum_response.png", dpi=300)
    fig.savefig(figures / "approach_spectrum_response.svg")
    plt.close(fig)

    metrics = read("approach_comparison.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for method, label in [
        ("none", "No filter"),
        ("ma3", "Three-point mean"),
        ("wide_lowpass", "Wide low-pass prototype"),
        ("multiband", "Multiband prototype"),
    ]:
        means = []
        for condition in range(5):
            rows = [
                r
                for r in metrics
                if r["method"] == method
                and int(r["condition_id"]) == condition
            ]
            means.append(
                (
                    float(rows[0]["target_snr_db"]),
                    float(np.mean([float(r["mse"]) for r in rows])),
                )
            )
        ax.semilogy(
            [v[0] for v in means],
            [v[1] for v in means],
            "o-",
            label=label,
            markersize=4,
        )
    ax.set(
        xlabel="Target input SNR (dB)",
        ylabel="Mean sample-domain MSE (NPY units squared)",
        title="Method-selection pilot: three paired design draws per level",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "approach_pilot_comparison.png", dpi=300)
    fig.savefig(figures / "approach_pilot_comparison.svg")
    plt.close(fig)


def main() -> None:
    """Measure reference examples and the R2.1 method-selection pilot.

    Returns:
        None. Write raw design data, checks, provenance and figures.
    """
    design = json.loads((ROOT / "design.json").read_text(encoding="utf-8"))
    results = ROOT / "results"
    figures = ROOT / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    original = np.load(PACKAGE / "reference_signal.npy")
    x = np.load(PACKAGE / "stage1/r12/results/selected_samples.npy")
    fs = design["sample_rate_hz"]
    n = len(x)
    if not np.array_equal(x, original[::3]):
        raise ValueError("R2 input differs from the selected R1.2 samples.")
    if n != 32000 or not np.all(np.isfinite(x)):
        raise ValueError("Unexpected Stage 1 input shape or values.")
    with (PACKAGE / "stage1/r11/results/components.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        priors = np.array(
            [float(r["frequency_hz"]) for r in csv.DictReader(stream)]
        )
    power = float(np.mean(x**2))
    u = float(make_rng((0, 0)).random())
    lower, upper = design["personal_snr_range_db"]
    personal_snr = lower + (upper - lower) * u
    conditions = [
        {"condition_id": i, "target_snr_db": snr, "role": "pilot"}
        for i, snr in zip(
            design["pilot_condition_ids"], design["pilot_snr_db"]
        )
    ]
    conditions.append(
        {
            "condition_id": design["personal_condition_id"],
            "target_snr_db": personal_snr,
            "role": "personal_design",
        }
    )
    write_csv(results / "condition_registry.csv", conditions)
    anchor_path = (1, 2, 0, design["personal_condition_id"], 0)
    anchor_y, _, _ = add_gaussian_noise(x, personal_snr, make_rng(anchor_path))
    frequency = np.fft.rfftfreq(n, 1 / fs)
    anchor_spectrum = np.fft.rfft(anchor_y)
    peak_rows = []
    for prior in priors:
        bins = np.flatnonzero(
            np.abs(frequency - prior) <= design["peak_search_half_width_hz"]
        )
        k = int(bins[np.argmax(np.abs(anchor_spectrum[bins]))])
        peak_rows.append(
            {
                "prior_frequency_hz": prior,
                "measured_bin": k,
                "measured_frequency_hz": frequency[k],
                "measured_noisy_amplitude": 2 * abs(anchor_spectrum[k]) / n,
            }
        )
    write_csv(results / "corrupted_peaks.csv", peak_rows)
    centres = [r["measured_frequency_hz"] for r in peak_rows]
    full_frequency = np.fft.fftfreq(n, 1 / fs)
    band_template = np.zeros(n)
    for centre in centres:
        band_template[
            np.abs(np.abs(full_frequency) - centre)
            <= design["prototype_band_half_width_hz"]
        ] = 1
    low_template = (
        np.abs(full_frequency) <= design["prototype_lowpass_edge_hz"]
    ).astype(float)
    length = design["prototype_kernel_length"]
    kernels = {
        "ma3": np.ones(3) / 3,
        "wide_lowpass": prototype_kernel(low_template, length),
        "multiband": prototype_kernel(band_template, length),
    }
    write_csv(
        results / "frequency_template.csv",
        [
            {
                "bin": k,
                "signed_frequency_hz": f,
                "multiband_template": band_template[k],
                "lowpass_template": low_template[k],
            }
            for k, f in enumerate(full_frequency)
        ],
    )
    coefficient_rows = []
    response_rows = []
    clean_rows = []
    margin = (length - 1) // 2
    edge = np.r_[np.arange(margin), np.arange(n - margin, n)]
    interior = np.arange(margin, n - margin)
    for method, h in kernels.items():
        delay = (len(h) - 1) // 2
        coefficient_rows.extend(
            [
                {
                    "method": method,
                    "index": m,
                    "centred_lag": m - delay,
                    "coefficient": value,
                }
                for m, value in enumerate(h)
            ]
        )
        response = np.fft.rfft(h, n)
        aligned = response * np.exp(2j * np.pi * frequency * delay / fs)
        response_rows.extend(
            [
                {
                    "method": method,
                    "frequency_hz": f,
                    "real": aligned[k].real,
                    "imag": aligned[k].imag,
                    "magnitude": abs(aligned[k]),
                    "phase_rad": np.angle(aligned[k]),
                }
                for k, f in enumerate(frequency)
            ]
        )
        error = centred_filter(x, h) - x
        clean_rows.append(
            {
                "method": method,
                "kernel_length": len(h),
                "delay_samples": delay,
                "noise_gain_interior": float(h @ h),
                "clean_full_mse": float(np.mean(error**2)),
                "clean_edge_mse": float(np.mean(error[edge] ** 2)),
                "clean_interior_mse": float(np.mean(error[interior] ** 2)),
            }
        )
    write_csv(results / "prototype_kernels.csv", coefficient_rows)
    write_csv(results / "prototype_response.csv", response_rows)
    write_csv(results / "clean_filter_distortion.csv", clean_rows)

    noise_rows, comparison, components, manifests, noise_draws = (
        [],
        [],
        [],
        [],
        [],
    )
    factors = np.full(n // 2 + 1, 2.0)
    factors[[0, -1]] = 1
    clean_spectrum = np.fft.rfft(x)
    anchor_outputs = {}
    max_equivalence_error = 0.0
    for condition in conditions:
        c = condition["condition_id"]
        snr = condition["target_snr_db"]
        for repeat in range(design["design_repeats"]):
            path = (1, 2, design["phase_id"], c, repeat)
            rng = make_rng(path)
            initial_state = rng.bit_generator.state
            y, noise, variance = add_gaussian_noise(x, snr, rng)
            run_id = f"R2-design-c{c}-r{repeat}"
            noise_rows.append(
                {
                    "run_id": run_id,
                    "condition_id": c,
                    "repeat_id": repeat,
                    "noise_array_row": len(noise_draws),
                    "target_snr_db": snr,
                    "noise_variance_model": variance,
                    "noise_mean": float(np.mean(noise)),
                    "noise_mean_square": float(np.mean(noise**2)),
                    "realised_snr_db": 10
                    * np.log10(power / np.mean(noise**2)),
                }
            )
            noise_draws.append(noise)
            manifests.append(
                {
                    "run_id": run_id,
                    "root_seed": 2470964,
                    "spawn_key": list(path),
                    "pool_size": 4,
                    "bit_generator": "PCG64",
                    "initial_state": initial_state,
                    "draw": "normal(0.0, sqrt(variance), (32000,))",
                    "dtype": "float64",
                    "target_snr_db": snr,
                    "noise_sha256": hashlib.sha256(
                        noise.tobytes()
                    ).hexdigest(),
                    "contaminated_sha256": hashlib.sha256(
                        y.tobytes()
                    ).hexdigest(),
                    "pairing_group": run_id,
                }
            )
            outputs = {"none": y}
            for method, h in kernels.items():
                outputs[method] = centred_filter(y, h)
                direct = np.convolve(y, h, mode="full")
                transformed = linear_fft_convolution(y, h)
                max_equivalence_error = max(
                    max_equivalence_error,
                    float(np.max(np.abs(direct - transformed))),
                )
            for method, output in outputs.items():
                error = output - x
                mse = float(np.mean(error**2))
                comparison.append(
                    {
                        "run_id": run_id,
                        "condition_id": c,
                        "repeat_id": repeat,
                        "method": method,
                        "target_snr_db": snr,
                        "mse": mse,
                        "rmse": np.sqrt(mse),
                        "nmse": mse / power,
                        "error_snr_db": 10 * np.log10(power / mse),
                        "edge_mse": float(np.mean(error[edge] ** 2)),
                        "interior_mse": float(np.mean(error[interior] ** 2)),
                        "baseline_mse": float(np.mean(noise**2)),
                        "gain_db": 10 * np.log10(np.mean(noise**2) / mse),
                    }
                )
                transformed = np.fft.rfft(output)
                for prior in priors:
                    k = int(round(prior * n / fs))
                    ratio = transformed[k] / clean_spectrum[k]
                    components.append(
                        {
                            "run_id": run_id,
                            "method": method,
                            "frequency_hz": prior,
                            "amplitude": 2 * abs(transformed[k]) / n,
                            "amplitude_ratio": abs(ratio),
                            "phase_error_rad": np.angle(ratio),
                        }
                    )
            if c == design["personal_condition_id"] and repeat == 0:
                anchor_outputs = outputs
                write_csv(
                    results / "sample_records.csv",
                    [
                        {
                            "sample": i,
                            "time_s": i / fs,
                            "clean": x[i],
                            "noise": noise[i],
                            "noisy": y[i],
                            "ma3": outputs["ma3"][i],
                            "wide_lowpass": outputs["wide_lowpass"][i],
                            "multiband": outputs["multiband"][i],
                        }
                        for i in range(n)
                    ],
                )
    np.save(results / "noise_draws.npy", np.stack(noise_draws))
    write_csv(results / "noise_trials.csv", noise_rows)
    write_csv(results / "approach_comparison.csv", comparison)
    write_csv(results / "component_retention.csv", components)
    anchor_transforms = {
        method: np.fft.rfft(output)
        for method, output in anchor_outputs.items()
    }
    write_csv(
        results / "pilot_spectrum.csv",
        [
            {
                "frequency_hz": f,
                "clean_amplitude": factors[k] * abs(clean_spectrum[k]) / n,
                "noisy_amplitude": factors[k]
                * abs(anchor_transforms["none"][k])
                / n,
                "noisy_power_per_bin": factors[k]
                * abs(anchor_transforms["none"][k]) ** 2
                / n**2,
            }
            for k, f in enumerate(frequency)
        ],
    )

    checks = []

    def check(name: str, error: float, tolerance: float) -> None:
        """Record a comparison with an independent expected result.

        Args:
            name: Meaning of the comparison.
            error: Nonnegative measured deviation.
            tolerance: Maximum permitted deviation.

        Returns:
            None. Save a check record and raise on failure.
        """
        checks.append(
            {
                "check": name,
                "error": error,
                "tolerance": tolerance,
                "passed": error <= tolerance,
            }
        )
        if error > tolerance:
            raise ValueError(name)

    a = np.array([2.0, 1.0])
    b = np.array([1.0, 2.0, 1.0])
    check(
        "convolution example hand convolution",
        float(
            np.max(
                np.abs(linear_fft_convolution(a, b) - np.array([2, 5, 4, 1]))
            )
        ),
        1e-12,
    )
    a = np.array([2.0, 4.0, 1.0, 0.0])
    b = np.array([3.0, 1.0, 2.0, 0.0])
    circular = np.array(
        [sum(a[m] * b[(i - m) % 4] for m in range(4)) for i in range(4)]
    )
    check(
        "DFT example direct circular expected values",
        float(np.max(np.abs(circular - np.array([8, 14, 11, 9])))),
        1e-12,
    )
    check(
        "DFT example circular convolution theorem",
        float(
            np.max(
                np.abs(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)) - circular)
            )
        ),
        1e-12,
    )
    a = np.array([1.0, 2.0, 1.0])
    b = np.array([2.0, 0.0, 2.0])
    correct = linear_fft_convolution(a, b)
    bad = np.fft.ifft(np.fft.fft(a, 4) * np.fft.fft(b, 4)).real
    check(
        "DFT example zero-padded linear expected values",
        float(np.max(np.abs(correct - np.array([2, 4, 4, 4, 2])))),
        1e-12,
    )
    check(
        "DFT example insufficient-padding expected wrap",
        float(np.max(np.abs(bad - np.array([4, 4, 4, 4])))),
        1e-12,
    )
    write_csv(
        results / "lab_convolution_examples.csv",
        [
            {"case": case, "sample": i, "value": value}
            for case, values in [
                ("linear_full", correct),
                ("insufficient_padding", bad),
            ]
            for i, value in enumerate(values)
        ],
    )
    check("R1.2 input unchanged", float(np.max(np.abs(x - original[::3]))), 0)
    check(
        "All full-record direct/FFT convolutions agree",
        max_equivalence_error,
        1e-11,
    )
    check(
        "Anchor leaf replays anchor draw",
        float(np.max(np.abs(anchor_y - anchor_outputs["none"]))),
        0,
    )
    check(
        "Contaminated peaks agree with declared R1 priors",
        float(np.max(np.abs(np.array(centres) - priors))),
        fs / n,
    )
    for method, h in kernels.items():
        check(
            method + " symmetric kernel",
            float(np.max(np.abs(h - h[::-1]))),
            1e-13,
        )
    check(
        "Identity filter preserves input",
        float(np.max(np.abs(centred_filter(x, np.ones(1)) - x))),
        1e-12,
    )
    check(
        "Zero filter produces zero",
        float(np.max(np.abs(centred_filter(x, np.zeros(1))))),
        0,
    )
    check(
        "Anchor Parseval mean square",
        abs(
            float(np.mean(anchor_y**2))
            - float(np.sum(factors * abs(anchor_spectrum) ** 2) / n**2)
        ),
        1e-12,
    )
    check(
        "MA3 six-kilohertz signed gain",
        abs(
            float(
                np.real(
                    np.sum(
                        kernels["ma3"]
                        * np.exp(-2j * np.pi * 6000 * (np.arange(3) - 1) / fs)
                    )
                )
            )
            - (1 - np.sqrt(2)) / 3
        ),
        1e-12,
    )
    write_csv(results / "checks.csv", checks)
    summary = {
        "input_power": power,
        "sample_count": n,
        "sample_rate_hz": fs,
        "parameter_uniform_draw": u,
        "personal_snr_db": personal_snr,
        "prototype_length": length,
        "prototype_half_width_hz": design["prototype_band_half_width_hz"],
        "full_length": n + length - 1,
        "fft_length": 1 << (n + length - 2).bit_length(),
        "max_direct_fft_error": max_equivalence_error,
        "design_noise_records": len(noise_rows),
        "check_count": len(checks),
    }
    write_csv(
        results / "summary.csv",
        [{"metric": key, "value": value} for key, value in summary.items()],
    )
    manifest = {
        "scope": "R2.1 design phase only; no held-out validation",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "root_seed": 2470964,
        "parameter_spawn_key": [0, 0],
        "snr_range_db": [lower, upper],
        "parameter_uniform_draw": u,
        "personal_snr_db": personal_snr,
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "design_sha256": hashlib.sha256(
            (ROOT / "design.json").read_bytes()
        ).hexdigest(),
        "reference_sha256": hashlib.sha256(
            (PACKAGE / "reference_signal.npy").read_bytes()
        ).hexdigest(),
        "selected_samples_sha256": hashlib.sha256(
            (PACKAGE / "stage1/r12/results/selected_samples.npy").read_bytes()
        ).hexdigest(),
        "prototype_design_source": "Personal design record, repeat 0",
        "prior": "R1 frequencies; search each +/-10 Hz in noisy spectrum",
        "prototype_parameters_final": False,
        "runs": manifests,
    }
    (results / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    make_figures(results, figures)
    print(
        f"R2.1: {len(checks)} checks passed; {len(noise_rows)} design "
        f"noise records; personal SNR {personal_snr:.9f} dB."
    )


if __name__ == "__main__":
    main()
