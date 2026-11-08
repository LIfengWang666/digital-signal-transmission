"""Select R2.2 FIR parameters from contaminated spectra and design records.

The search uses the reference DFT/IDFT, convolution and three window example windows.
No phase-1 data, quantisation or bit-error channel is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from stage2.r21.filtering_approach import (  # noqa: E402
    centred_filter,
    linear_fft_convolution,
    write_csv,
)

FloatArray = NDArray[np.float64]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one saved design CSV.

    Args:
        path: Existing UTF-8 CSV.

    Returns:
        Header-keyed records.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def window_values(name: str, length: int) -> FloatArray:
    """Use the three symmetric window definitions studied in window example.

    Args:
        name: rectangular, hann or hamming.
        length: Number of kernel coefficients.

    Returns:
        A real window without gain renormalisation.
    """
    if name == "rectangular":
        return np.ones(length)
    if name == "hann":
        return np.hanning(length)
    if name == "hamming":
        return np.hamming(length)
    raise ValueError("Unknown registered window.")


def frequency_kernel(
    centres: FloatArray,
    half_width: float,
    length: int,
    window: str,
    sample_rate: float,
    grid_length: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Construct a finite symmetric kernel from an even frequency template.

    Args:
        centres: Frequencies measured in the contaminated design spectrum.
        half_width: Half-width of each positive/negative passband in Hz.
        length: Odd number of retained impulse-response coefficients.
        window: Registered symmetric kernel taper.
        sample_rate: Samples per second.
        grid_length: Number of real samples in the design record.

    Returns:
        Finite FIR coefficients, full binary template, and untapered
        centred impulse samples.
    """
    if length % 2 != 1 or length > grid_length:
        raise ValueError("Use an odd finite kernel within the design grid.")
    f = np.fft.fftfreq(grid_length, 1 / sample_rate)
    template = np.zeros(grid_length)
    for centre in centres:
        template[np.abs(np.abs(f) - centre) <= half_width] = 1
    impulse = np.fft.ifft(template)
    if np.max(np.abs(impulse.imag)) > 1e-13:
        raise ValueError("The even real template must have a real IDFT.")
    delay = (length - 1) // 2
    lags = np.arange(-delay, delay + 1)
    untapered = impulse.real[lags % grid_length]
    return (untapered * window_values(window, length), template, untapered)


def choose_candidate(
    candidates: list[dict], fraction: float, order: list[str]
) -> dict:
    """Apply the declared feasible-set and shortest-near-minimum rule.

    Args:
        candidates: Measured candidate records.
        fraction: Relative score allowance above the strict minimum.
        order: Stable window IDs used only for residual ties.

    Returns:
        One selected record; raise if no feasible candidate exists.
    """
    feasible = [row for row in candidates if row["feasible"]]
    if not feasible:
        raise ValueError("No candidate meets the declared gain constraint.")
    best = min(row["score"] for row in feasible)
    shortlist = [
        row for row in feasible if row["score"] <= best * (1 + fraction)
    ]
    return min(
        shortlist,
        key=lambda row: (
            row["length"],
            row["score"],
            row["half_width_hz"],
            order.index(row["window"]),
        ),
    )


def measured_width(
    frequency: FloatArray, amplitude: FloatArray, centre: float
) -> float:
    """Measure a noisy local peak's literal -3 dB full width.

    Args:
        frequency: Dense samples of the same finite record's DTFT.
        amplitude: Consistently scaled positive amplitudes.
        centre: Prior local peak centre in Hz.

    Returns:
        Width between linearly interpolated -3.000 dB crossings in Hz.
    """
    local = np.flatnonzero(np.abs(frequency - centre) <= 1)
    peak = int(local[np.argmax(amplitude[local])])
    threshold = amplitude[peak] * 10 ** (-3 / 20)
    left = peak
    right = peak
    while left > 0 and amplitude[left] > threshold:
        left -= 1
    while right < len(amplitude) - 1 and amplitude[right] > threshold:
        right += 1
    if left == 0 or right == len(amplitude) - 1:
        raise ValueError("The local peak has no enclosing -3 dB crossings.")
    f_left = np.interp(
        threshold,
        amplitude[[left, left + 1]],
        frequency[[left, left + 1]],
    )
    f_right = np.interp(
        threshold,
        amplitude[[right, right - 1]],
        frequency[[right, right - 1]],
    )
    return float(f_right - f_left)


def main() -> None:
    """Measure spectra, scan parameters, freeze a filter and save evidence.

    Returns:
        None. Write design CSV, coefficients, manifest and checks.
    """
    design = json.loads((ROOT / "design.json").read_text(encoding="utf-8"))
    previous = PACKAGE / "stage2/r21/results"
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    x = np.load(PACKAGE / "stage1/r12/results/selected_samples.npy")
    original = np.load(PACKAGE / "reference_signal.npy")
    if not np.array_equal(x, original[::3]):
        raise ValueError("R2.2 input differs from the original R1.2 samples.")
    n = len(x)
    fs = design["sample_rate_hz"]
    noise = np.load(previous / "noise_draws.npy")
    noise_rows = read_csv(previous / "noise_trials.csv")
    source_manifest = json.loads(
        (previous / "seed_manifest.json").read_text(encoding="utf-8")
    )
    for i, run in enumerate(source_manifest["runs"]):
        if run["spawn_key"][2] != 0:
            raise ValueError("R2.2 may use only design-phase records.")
        if (
            hashlib.sha256(noise[i].tobytes()).hexdigest()
            != run["noise_sha256"]
        ):
            raise ValueError("Stored design noise has changed.")
    y = x[None, :] + noise
    centres = np.array(
        [
            float(row["measured_frequency_hz"])
            for row in read_csv(previous / "corrupted_peaks.csv")
        ]
    )
    conditions = np.array([int(row["condition_id"]) for row in noise_rows])
    objective_rows = np.isin(conditions, design["objective_condition_ids"])
    baseline = np.mean((y - x) ** 2, axis=1)
    power = float(np.mean(x**2))
    anchor_index = next(
        i
        for i, row in enumerate(noise_rows)
        if int(row["condition_id"]) == 5 and int(row["repeat_id"]) == 0
    )
    anchor = y[anchor_index]
    f = np.fft.rfftfreq(n, 1 / fs)
    factors = np.full(len(f), 2.0)
    factors[[0, -1]] = 1
    anchor_fft = np.fft.rfft(anchor)
    anchor_psd = factors * abs(anchor_fft) ** 2 / (fs * n)
    background_bins = (f > 0) & (f < fs / 2)
    for centre in centres:
        background_bins &= (
            np.abs(f - centre) > design["background_exclusion_hz"]
        )
    background = float(np.mean(anchor_psd[background_bins]))
    dense_length = n * design["analysis_fft_padding_factor"]
    dense_f = np.fft.rfftfreq(dense_length, 1 / fs)
    dense_factors = np.full(len(dense_f), 2.0)
    dense_factors[[0, -1]] = 1
    dense_amplitudes = {}
    psd_errors = []
    spectrum_rows = []
    local_rows = []
    for window in design["windows"]:
        w = window_values(window, n)
        transformed = np.fft.rfft(anchor * w)
        psd = factors * abs(transformed) ** 2 / (fs * np.sum(w**2))
        psd_errors.append(
            abs(
                float(np.sum(psd) * fs / n)
                - float(np.sum(anchor**2 * w**2) / np.sum(w**2))
            )
        )
        dense_amplitudes[window] = (
            dense_factors
            * abs(np.fft.rfft(anchor * w, dense_length))
            / np.sum(w)
        )
        spectrum_rows.extend(
            [
                {
                    "window": window,
                    "frequency_hz": value,
                    "amplitude": factors[k] * abs(transformed[k]) / np.sum(w),
                    "psd": psd[k],
                }
                for k, value in enumerate(f)
            ]
        )
        for centre in centres:
            bins = np.flatnonzero(np.abs(dense_f - centre) <= 10)
            local_rows.extend(
                [
                    {
                        "window": window,
                        "centre_hz": centre,
                        "offset_hz": dense_f[k] - centre,
                        "amplitude": dense_amplitudes[window][k],
                    }
                    for k in bins
                ]
            )
    write_csv(results / "corrupted_spectrum.csv", spectrum_rows)
    write_csv(results / "local_peak_spectra.csv", local_rows)
    spectral = []
    for centre in centres:
        k = int(round(centre * n / fs))
        spectral.append(
            {
                "centre_hz": centre,
                "rectangular_peak_psd": anchor_psd[k],
                "offband_mean_psd": background,
                "peak_to_background_db": 10
                * np.log10(anchor_psd[k] / background),
                **{
                    window
                    + "_width_3db_hz": measured_width(dense_f, values, centre)
                    for window, values in dense_amplitudes.items()
                },
            }
        )
    write_csv(results / "spectral_evidence.csv", spectral)

    offsets = np.arange(
        -design["gain_guard_hz"],
        design["gain_guard_hz"] + design["gain_guard_step_hz"] / 2,
        design["gain_guard_step_hz"],
    )
    guard_f = (centres[:, None] + offsets).ravel()
    basis_cache = {}
    transform_cache = {}
    clean_transform_cache = {}
    candidate_rows = []
    trial_rows = []
    tone_rows = []
    seen = set()
    fixed_edge = design["fixed_edge_samples"]
    interior_end = n - fixed_edge
    edge = np.r_[np.arange(fixed_edge), np.arange(n - fixed_edge, n)]
    decomposition_error = 0.0

    def evaluate(length: int, width: float, window: str, stage: str) -> None:
        """Measure a unique candidate on exactly the registered design data.

        Args:
            length: Odd finite kernel length.
            width: Passband half-width in Hz.
            window: Registered symmetric taper.
            stage: coarse or refined, retained in the data.

        Returns:
            None. Append summary, trial and tone records.
        """
        nonlocal decomposition_error
        key = (length, float(width), window)
        if key in seen:
            return
        seen.add(key)
        h, _, _ = frequency_kernel(centres, width, length, window, fs, n)
        delay = (length - 1) // 2
        lags = np.arange(-delay, delay + 1)
        if length not in basis_cache:
            basis_cache[length] = np.exp(
                -2j * np.pi * guard_f[:, None] * lags / fs
            )
        gains = basis_cache[length] @ h
        deviation = float(np.max(np.abs(np.abs(gains) - 1)))
        feasible = (
            deviation <= design["maximum_gain_deviation"]
            and np.all(gains.real > 0)
            and np.max(np.abs(gains.imag)) < 1e-12
        )
        full_length = n + length - 1
        fft_length = 1 << (full_length - 1).bit_length()
        if fft_length not in transform_cache:
            transform_cache[fft_length] = np.fft.fft(y, fft_length, axis=1)
            clean_transform_cache[fft_length] = np.fft.fft(x, fft_length)
        response = np.fft.fft(h, fft_length)
        full = np.fft.ifft(
            transform_cache[fft_length] * response[None, :], axis=1
        )
        if np.max(np.abs(full.imag)) > 1e-11:
            raise ValueError("A candidate generated a complex output.")
        end = delay + n
        output = full.real[:, delay:end]
        clean_full = np.fft.ifft(clean_transform_cache[fft_length] * response)
        clean_output = clean_full.real[delay:end]
        distortion = clean_output - x
        residual_noise = output - clean_output
        errors = output - x
        mse = np.mean(errors**2, axis=1)
        distortion_mse = float(np.mean(distortion**2))
        residual_mse = np.mean(residual_noise**2, axis=1)
        cross = 2 * np.mean(residual_noise * distortion, axis=1)
        decomposition_error = max(
            decomposition_error,
            float(np.max(np.abs(mse - distortion_mse - residual_mse - cross))),
        )
        effective_noise_gain = float(np.sum(h**2 * (1 - np.abs(lags) / n)))
        score = float(np.mean(mse[objective_rows] / baseline[objective_rows]))
        candidate_id = f"M{length}-B{width:g}-{window}"
        row = {
            "candidate_id": candidate_id,
            "stage": stage,
            "length": length,
            "half_width_hz": width,
            "window": window,
            "delay_samples": delay,
            "fft_length": fft_length,
            "gain_max_deviation": deviation,
            "feasible": bool(feasible),
            "score": score,
            "clean_full_mse": distortion_mse,
            "clean_fixed_edge_mse": float(np.mean(distortion[edge] ** 2)),
            "clean_fixed_interior_mse": float(
                np.mean(distortion[fixed_edge:interior_end] ** 2)
            ),
            "noise_gain_interior": float(h @ h),
            "noise_gain_full": effective_noise_gain,
        }
        candidate_rows.append(row)
        centre_gains = gains.reshape(len(centres), len(offsets))[
            :, len(offsets) // 2
        ]
        for i, centre in enumerate(centres):
            tone_rows.append(
                {
                    "candidate_id": candidate_id,
                    "frequency_hz": centre,
                    "signed_gain": centre_gains[i].real,
                    "gain_magnitude": abs(centre_gains[i]),
                    "phase_rad": np.angle(centre_gains[i]),
                }
            )
        for i, source in enumerate(noise_rows):
            trial_rows.append(
                {
                    "candidate_id": candidate_id,
                    "run_id": source["run_id"],
                    "condition_id": int(source["condition_id"]),
                    "repeat_id": int(source["repeat_id"]),
                    "target_snr_db": float(source["target_snr_db"]),
                    "used_in_objective": bool(objective_rows[i]),
                    "mse": mse[i],
                    "baseline_mse": baseline[i],
                    "mse_ratio": mse[i] / baseline[i],
                    "fixed_edge_mse": float(np.mean(errors[i, edge] ** 2)),
                    "fixed_interior_mse": float(
                        np.mean(errors[i, fixed_edge:interior_end] ** 2)
                    ),
                    "clean_distortion_mse": distortion_mse,
                    "residual_noise_mse": residual_mse[i],
                    "cross_term": cross[i],
                }
            )

    for length in design["coarse_lengths"]:
        for width in design["coarse_half_widths_hz"]:
            for window in design["windows"]:
                evaluate(length, width, window, "coarse")
    coarse_best = choose_candidate(candidate_rows, 0, design["window_order"])
    length_index = design["coarse_lengths"].index(coarse_best["length"])
    width_index = design["coarse_half_widths_hz"].index(
        coarse_best["half_width_hz"]
    )
    length_bounds = [
        design["coarse_lengths"][max(0, length_index - 1)],
        design["coarse_lengths"][
            min(len(design["coarse_lengths"]) - 1, length_index + 1)
        ],
    ]
    width_bounds = [
        design["coarse_half_widths_hz"][max(0, width_index - 1)],
        design["coarse_half_widths_hz"][
            min(len(design["coarse_half_widths_hz"]) - 1, width_index + 1)
        ],
    ]
    lengths = sorted(
        {
            int(2 * round((v - 1) / 2) + 1)
            for v in np.linspace(
                *length_bounds, design["refined_length_count"]
            )
        }
    )
    widths = sorted(
        {
            float(round(v / (fs / n)) * (fs / n))
            for v in np.linspace(*width_bounds, design["refined_width_count"])
        }
    )
    for length in lengths:
        for width in widths:
            for window in design["windows"]:
                evaluate(length, width, window, "refined")
    selected = choose_candidate(
        candidate_rows,
        design["near_minimum_fraction"],
        design["window_order"],
    )
    strict = choose_candidate(candidate_rows, 0, design["window_order"])
    write_csv(results / "parameter_scan.csv", candidate_rows)
    write_csv(results / "candidate_trials.csv", trial_rows)
    write_csv(results / "candidate_tone_gains.csv", tone_rows)
    sensitivity = []
    for tolerance in [0, 0.005, 0.01, 0.05]:
        choice = choose_candidate(
            candidate_rows, tolerance, design["window_order"]
        )
        sensitivity.append(
            {
                "near_minimum_fraction": tolerance,
                "candidate_id": choice["candidate_id"],
                "length": choice["length"],
                "half_width_hz": choice["half_width_hz"],
                "window": choice["window"],
                "score": choice["score"],
            }
        )
    write_csv(results / "selection_sensitivity.csv", sensitivity)
    h, template, untapered = frequency_kernel(
        centres,
        selected["half_width_hz"],
        selected["length"],
        selected["window"],
        fs,
        n,
    )
    delay = selected["delay_samples"]
    taper = window_values(selected["window"], len(h))
    write_csv(
        results / "selected_coefficients.csv",
        [
            {
                "index": m,
                "centred_lag": m - delay,
                "untapered": untapered[m],
                "window": taper[m],
                "coefficient": value,
            }
            for m, value in enumerate(h)
        ],
    )
    np.save(results / "selected_coefficients.npy", h)
    response = np.fft.rfft(h, n)
    aligned = response * np.exp(2j * np.pi * f * delay / fs)
    write_csv(
        results / "selected_response.csv",
        [
            {
                "frequency_hz": value,
                "template_gain": template[k],
                "causal_real": response[k].real,
                "causal_imag": response[k].imag,
                "aligned_real": aligned[k].real,
                "aligned_imag": aligned[k].imag,
                "magnitude": abs(response[k]),
                "phase_rad": np.angle(aligned[k]),
            }
            for k, value in enumerate(f)
        ],
    )
    selected_rows = [
        row
        for row in trial_rows
        if row["candidate_id"] == selected["candidate_id"]
    ]
    prototype_rows = [
        row for row in trial_rows if row["candidate_id"] == "M513-B80-hann"
    ]
    write_csv(results / "selected_trials.csv", selected_rows)
    write_csv(results / "prototype_trials.csv", prototype_rows)
    selected_output = centred_filter(anchor, h)
    clean_output = centred_filter(x, h)
    old_coefficients = read_csv(previous / "prototype_kernels.csv")
    old_h = np.array(
        [
            float(row["coefficient"])
            for row in old_coefficients
            if row["method"] == "multiband"
        ]
    )
    prototype_response = np.fft.rfft(old_h, n)
    final_bands = [
        {
            "centre_hz": centre,
            "nominal_lower_hz": centre - selected["half_width_hz"],
            "nominal_upper_hz": centre + selected["half_width_hz"],
            "selected_width_3db_hz": measured_width(f, abs(response), centre),
            "prototype_width_3db_hz": measured_width(
                f, abs(prototype_response), centre
            ),
        }
        for centre in centres
    ]
    write_csv(results / "final_bands.csv", final_bands)
    prototype_output = centred_filter(anchor, old_h)
    prototype_clean_output = centred_filter(x, old_h)
    write_csv(
        results / "boundary_records.csv",
        [
            {
                "sample": i,
                "time_s": i / fs,
                "clean": x[i],
                "noisy": anchor[i],
                "selected": selected_output[i],
                "prototype": prototype_output[i],
                "selected_clean_error": clean_output[i] - x[i],
                "prototype_clean_error": prototype_clean_output[i] - x[i],
            }
            for i in range(n)
        ],
    )
    checks = []

    def check(name: str, error: float, tolerance: float) -> None:
        """Record an independent mathematical or provenance comparison.

        Args:
            name: Meaning of the check.
            error: Nonnegative discrepancy.
            tolerance: Maximum permitted discrepancy.

        Returns:
            None. Append a record and raise if it fails.
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

    check(
        "Windowed PSD integrates to the weighted mean square",
        max(psd_errors),
        1e-12,
    )
    check(
        "Finite-noise decomposition includes the single-record cross term",
        decomposition_error,
        1e-11,
    )
    check(
        "Selected kernel is symmetric",
        float(np.max(np.abs(h - h[::-1]))),
        1e-13,
    )
    check(
        "Selected full FFT and direct linear convolution agree",
        float(
            np.max(
                np.abs(
                    linear_fft_convolution(anchor, h)
                    - np.convolve(anchor, h, mode="full")
                )
            )
        ),
        1e-11,
    )
    direct_errors = []
    for m in [delay, delay + 1, delay + 7]:
        lag = m - delay
        expected = (
            np.sum(template * np.exp(2j * np.pi * np.arange(n) * lag / n))
            / n
            * taper[m]
        )
        direct_errors.append(abs(h[m] - expected))
    check(
        "Three coefficients match the explicit IDFT sum",
        float(max(direct_errors)),
        1e-12,
    )
    toy_h = np.array([0.2, 0.6, 0.2])
    toy_n = 9
    toy_end = toy_n + 1
    energy = 0.0
    for index in range(toy_n):
        unit = np.zeros(toy_n)
        unit[index] = 1
        full = np.convolve(unit, toy_h)
        energy += float(np.sum(full[1:toy_end] ** 2)) / toy_n
    expected_gain = float(
        np.sum(toy_h**2 * (1 - np.abs(np.arange(-1, 2)) / toy_n))
    )
    check(
        "Finite-boundary noise gain matches an impulse-basis trace",
        abs(energy - expected_gain),
        1e-14,
    )
    for name in ["hann", "hamming"]:
        alpha = 0.5 if name == "hann" else 0.54
        expected = alpha - (1 - alpha) * np.cos(
            2 * np.pi * np.arange(len(h)) / (len(h) - 1)
        )
        check(
            name + " agrees with the written symmetric formula",
            float(np.max(np.abs(window_values(name, len(h)) - expected))),
            1e-14,
        )
    dense_guard = np.concatenate(
        [centre + np.linspace(-2, 2, 129) for centre in centres]
    )
    dense_gain = (
        np.exp(
            -2j
            * np.pi
            * dense_guard[:, None]
            * np.arange(-delay, delay + 1)
            / fs
        )
        @ h
    )
    dense_deviation = float(np.max(np.abs(abs(dense_gain) - 1)))
    check(
        "Selected gain constraint also holds on a denser guard grid",
        dense_deviation,
        design["maximum_gain_deviation"],
    )
    check(
        "Selected score obeys the declared one-percent allowance",
        max(0, selected["score"] / strict["score"] - 1),
        design["near_minimum_fraction"],
    )
    write_csv(results / "checks.csv", checks)
    frozen = {
        "status": "Frozen for R2.3; chosen only from phase-0 design data",
        "selected": selected,
        "strict_minimum": strict,
        "coarse_best": coarse_best,
        "refined_lengths": lengths,
        "refined_half_widths_hz": widths,
        "centres_hz": centres.tolist(),
        "sample_rate_hz": fs,
        "record_samples": n,
        "kernel_origin": delay,
        "boundary": design["boundary"],
        "gain_guard_dense_max_deviation": dense_deviation,
        "coefficient_sha256": hashlib.sha256(
            (results / "selected_coefficients.npy").read_bytes()
        ).hexdigest(),
        "design_sha256": hashlib.sha256(
            (ROOT / "design.json").read_bytes()
        ).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_noise_sha256": hashlib.sha256(
            (previous / "noise_draws.npy").read_bytes()
        ).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            (previous / "seed_manifest.json").read_bytes()
        ).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "source_input_sha256": {
            name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
            for name in [
                "reference_signal.npy",
                "stage1/r12/results/selected_samples.npy",
                "stage2/r21/results/corrupted_peaks.csv",
                "stage2/r21/results/noise_trials.csv",
                "stage2/r21/results/prototype_kernels.csv",
            ]
        },
        "candidate_count": len(candidate_rows),
        "feasible_count": sum(row["feasible"] for row in candidate_rows),
        "design_records": len(noise_rows),
        "phase_1_used": False,
        "source_spawn_keys": [
            run["spawn_key"] for run in source_manifest["runs"]
        ],
        "scope": "Bounded coarse/refined family; no global optimum claim",
    }
    (results / "filter_freeze.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        results / "summary.csv",
        [
            {"metric": key, "value": value}
            for key, value in {
                "candidate_count": len(candidate_rows),
                "feasible_count": frozen["feasible_count"],
                "selected_length": len(h),
                "selected_half_width_hz": selected["half_width_hz"],
                "selected_score": selected["score"],
                "strict_minimum_score": strict["score"],
                "offband_mean_psd": background,
                "selected_clean_mse": selected["clean_full_mse"],
                "prototype_clean_mse": next(
                    row["clean_full_mse"]
                    for row in candidate_rows
                    if row["candidate_id"] == "M513-B80-hann"
                ),
                "check_count": len(checks),
            }.items()
        ],
    )
    print(
        f"R2.2: {len(candidate_rows)} candidates; selected "
        f"{selected['candidate_id']}; J={selected['score']:.9f}; "
        f"{len(checks)} checks passed."
    )


if __name__ == "__main__":
    main()
