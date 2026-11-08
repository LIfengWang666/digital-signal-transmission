"""Audit all Stage 2 candidates and validation records independently.

This checks the frozen investigation. It never selects a replacement filter,
changes the noise protocol, or writes the stored experimental results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FloatArray = NDArray[np.float64]


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read an existing measured table.

    Args:
        path: A UTF-8 CSV with a header.

    Returns:
        Records keyed by their original column names.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write an audit table separately from the experimental results.

    Args:
        path: Audit destination.
        rows: Nonempty records with matching fields.

    Returns:
        None. Write a stable header and all records.
    """
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analytic_kernel(
    length: int,
    half_width: float,
    window: str,
    centres: FloatArray,
    count: int,
    rate: float,
) -> FloatArray:
    """Evaluate the symmetric mask's IDFT by a finite cosine-series sum.

    Args:
        length: Odd retained coefficient count.
        half_width: Nominal passband half-width in Hz.
        window: One of the three declared symmetric tapers.
        centres: Four positive centres, all on native DFT bins.
        count: Original DFT template length.
        rate: Original sample rate in Hz.

    Returns:
        Coefficients computed without the production IFFT constructor.
    """
    delay = (length - 1) // 2
    lags = np.arange(-delay, delay + 1)
    half_bins = int(round(half_width * count / rate))
    band_bins = 2 * half_bins + 1
    # sin(B*pi*l/N)/sin(pi*l/N), including its limit B at l=0.
    geometric_sum = (
        band_bins * np.sinc(band_bins * lags / count) / np.sinc(lags / count)
    )
    carrier_sum = np.sum(
        np.cos(2 * np.pi * centres[:, None] * lags / rate), axis=0
    )
    theta = 2 * np.pi * np.arange(length) / (length - 1)
    if window == "rectangular":
        taper = np.ones(length)
    elif window == "hann":
        taper = 0.5 - 0.5 * np.cos(theta)
    elif window == "hamming":
        taper = 0.54 - 0.46 * np.cos(theta)
    else:
        raise ValueError("Unregistered window.")
    return 2 * geometric_sum * carrier_sum * taper / count


def main() -> None:
    """Recompute the selected investigation with independent numerical paths.

    Returns:
        None. Write audit CSV/JSON and raise if any comparison fails.
    """
    r21 = ROOT / "stage2/r21/results"
    r22 = ROOT / "stage2/r22/results"
    r23 = ROOT / "stage2/r23/results"
    metadata = ROOT / "metadata"
    frozen = json.loads((r22 / "filter_freeze.json").read_text())
    design = json.loads((ROOT / "stage2/r22/design.json").read_text())
    reference_path = ROOT / "reference_signal.npy"
    reference = np.load(reference_path)
    x = np.load(ROOT / "stage1/r12/results/selected_samples.npy")
    h = np.load(r22 / "selected_coefficients.npy")
    count = len(x)
    rate = frozen["sample_rate_hz"]
    centres = np.array(frozen["centres_hz"])
    checks = []

    def check(name: str, error: float, tolerance: float) -> None:
        """Record a finite discrepancy and enforce its numerical bound.

        Args:
            name: Meaning of the independent comparison.
            error: Absolute nonnegative discrepancy or failure count.
            tolerance: Largest accepted value.

        Returns:
            None. Append evidence and fail on inconsistency.
        """
        passed = bool(np.isfinite(error) and 0 <= error <= tolerance)
        checks.append(
            {
                "check": name,
                "error": float(error),
                "tolerance": tolerance,
                "passed": passed,
            }
        )
        if not passed:
            raise AssertionError(name)

    expected_reference = (
        "2f62c84fb0e11586373b2dc0d92f180206d15d4a77098046c9c43de290657095"
    )
    expected_coefficients = (
        "aa155baf5455d627aa4df47373cad6c6d254fa1359020dc10895fcc28e89dec8"
    )
    check(
        "Original reference SHA256",
        float(
            hashlib.sha256(reference_path.read_bytes()).hexdigest()
            != expected_reference
        ),
        0,
    )
    check(
        "Exact R1.2 decimated input identity",
        float(not np.array_equal(x, reference[::3])),
        0,
    )
    coefficient_hash = hashlib.sha256(
        (r22 / "selected_coefficients.npy").read_bytes()
    ).hexdigest()
    check(
        "Pre-validation frozen coefficients unchanged",
        float(coefficient_hash != expected_coefficients),
        0,
    )
    check(
        "Audited gain guard matches the declared protocol",
        float(
            design["maximum_gain_deviation"] != 0.01
            or design["gain_guard_hz"] != 2
            or design["gain_guard_step_hz"] != 0.5
            or design["near_minimum_fraction"] != 0.01
        ),
        0,
    )
    noise = np.load(r21 / "noise_draws.npy")
    noise_rows = read_rows(r21 / "noise_trials.csv")
    design_manifest = json.loads((r21 / "seed_manifest.json").read_text())
    design_paths = {tuple(run["spawn_key"]) for run in design_manifest["runs"]}
    check(
        "18 distinct phase-0 design paths",
        float(len(design_paths) != 18 or any(p[2] != 0 for p in design_paths)),
        0,
    )
    y = x[None, :] + noise
    anchor_index = next(
        index
        for index, row in enumerate(noise_rows)
        if row["condition_id"] == "5" and row["repeat_id"] == "0"
    )
    anchor = y[anchor_index]
    f = np.fft.rfftfreq(count, 1 / rate)
    factors = np.full(len(f), 2.0)
    factors[[0, -1]] = 1
    native = read_rows(r22 / "corrupted_spectrum.csv")
    local = read_rows(r22 / "local_peak_spectra.csv")
    spectrum_error = 0.0
    local_error = 0.0
    integral_error = 0.0
    dense_count = 16 * count
    dense_factors = np.full(dense_count // 2 + 1, 2.0)
    dense_factors[[0, -1]] = 1
    theta = 2 * np.pi * np.arange(count) / (count - 1)
    for window, values in [
        ("rectangular", np.ones(count)),
        ("hann", 0.5 - 0.5 * np.cos(theta)),
        ("hamming", 0.54 - 0.46 * np.cos(theta)),
    ]:
        for transform_count in [count, dense_count]:
            transformed = np.fft.rfft(anchor * values, transform_count)
            weights = factors if transform_count == count else dense_factors
            amplitude = weights * abs(transformed) / np.sum(values)
            psd = weights * abs(transformed) ** 2 / (rate * np.sum(values**2))
            integral_error = max(
                integral_error,
                abs(
                    float(np.sum(psd) * rate / transform_count)
                    - float(np.sum(anchor**2 * values**2) / np.sum(values**2))
                ),
            )
            rows = [
                row
                for row in (native if transform_count == count else local)
                if row["window"] == window
            ]
            for row in rows:
                frequency = (
                    float(row["frequency_hz"])
                    if transform_count == count
                    else float(row["centre_hz"]) + float(row["offset_hz"])
                )
                k = int(round(frequency * transform_count / rate))
                error = abs(float(row["amplitude"]) - amplitude[k])
                if transform_count == count:
                    spectrum_error = max(
                        spectrum_error, error, abs(float(row["psd"]) - psd[k])
                    )
                else:
                    local_error = max(local_error, error)
    check("All native windowed amplitudes and PSDs", spectrum_error, 1e-12)
    check("All saved padded local-peak amplitudes", local_error, 1e-12)
    check("PSD Parseval integrals for N and 16N", integral_error, 1e-12)
    # Native peak locations are measured, not replaced by the reference tones.
    measured = read_rows(r21 / "corrupted_peaks.csv")
    transform = np.fft.rfft(anchor)
    peak_error = 0.0
    for row in measured:
        prior = float(row["prior_frequency_hz"])
        indices = np.flatnonzero(abs(f - prior) <= 10)
        k = indices[np.argmax(abs(transform[indices]))]
        peak_error = max(
            peak_error, abs(f[k] - float(row["measured_frequency_hz"]))
        )
    check("Four corrupted-spectrum peak searches", peak_error, 0)

    # Audit all candidates using a cosine-series kernel and real FFTs at
    # K=64000, different from the production complex power-of-two transforms.
    fft_length = 2 * count
    input_fft = np.fft.rfft(y, fft_length, axis=1)
    clean_fft = np.fft.rfft(x, fft_length)
    baseline = np.mean((y - x) ** 2, axis=1)
    objective = np.array(
        [int(row["condition_id"]) in [1, 2, 3, 4, 5] for row in noise_rows]
    )
    candidates = read_rows(r22 / "parameter_scan.csv")
    trial_groups = {}
    for row in read_rows(r22 / "candidate_trials.csv"):
        trial_groups.setdefault(row["candidate_id"], {})[row["run_id"]] = row
    check(
        "372 candidates each retain all 18 design records",
        float(
            len(candidates) != 372
            or len(trial_groups) != 372
            or any(len(group) != 18 for group in trial_groups.values())
        ),
        0,
    )
    candidate_checks = []
    max_candidate_mse_error = 0.0
    max_candidate_score_error = 0.0
    max_distortion_error = 0.0
    max_gain_error = 0.0
    feasible_mismatches = 0
    selected_coefficient_error = 0.0
    offsets = np.arange(-2, 2.01, 0.5)
    guards = (centres[:, None] + offsets).ravel()
    for row in candidates:
        length = int(row["length"])
        candidate = analytic_kernel(
            length,
            float(row["half_width_hz"]),
            row["window"],
            centres,
            count,
            rate,
        )
        delay = (length - 1) // 2
        end = delay + count
        response = np.fft.rfft(candidate, fft_length)
        full = np.fft.irfft(input_fft * response, fft_length, axis=1)
        outputs = full[:, delay:end]
        errors = outputs - x
        mse = np.mean(errors**2, axis=1)
        score = float(np.mean(mse[objective] / baseline[objective]))
        clean_full = np.fft.irfft(clean_fft * response, fft_length)
        clean_error = clean_full[delay:end] - x
        clean_mse = float(np.mean(clean_error**2))
        lags = np.arange(-delay, delay + 1)
        gain = float(np.sum(candidate**2 * (1 - abs(lags) / count)))
        guards_gain = (
            np.exp(-2j * np.pi * guards[:, None] * lags / rate) @ candidate
        )
        guard_deviation = float(np.max(abs(abs(guards_gain) - 1)))
        feasible = bool(
            guard_deviation <= 0.01
            and np.all(guards_gain.real > 0)
            and np.max(abs(guards_gain.imag)) < 1e-12
        )
        feasible_mismatches += feasible != (row["feasible"] == "True")
        for index, source in enumerate(noise_rows):
            saved = trial_groups[row["candidate_id"]][source["run_id"]]
            max_candidate_mse_error = max(
                max_candidate_mse_error, abs(mse[index] - float(saved["mse"]))
            )
        max_candidate_score_error = max(
            max_candidate_score_error, abs(score - float(row["score"]))
        )
        max_distortion_error = max(
            max_distortion_error, abs(clean_mse - float(row["clean_full_mse"]))
        )
        max_gain_error = max(
            max_gain_error, abs(gain - float(row["noise_gain_full"]))
        )
        if row["candidate_id"] == frozen["selected"]["candidate_id"]:
            selected_coefficient_error = float(np.max(abs(candidate - h)))
        candidate_checks.append(
            {
                "candidate_id": row["candidate_id"],
                "length": length,
                "window": row["window"],
                "half_width_hz": float(row["half_width_hz"]),
                "saved_score": float(row["score"]),
                "independent_score": score,
                "score_difference": score - float(row["score"]),
                "independent_feasible": feasible,
                "independent_clean_mse": clean_mse,
                "independent_finite_noise_gain": gain,
            }
        )
    check(
        "Frozen kernel agrees with the independent cosine-series sum",
        selected_coefficient_error,
        1e-13,
    )
    check(
        "All 6696 candidate-record full MSE values",
        max_candidate_mse_error,
        1e-11,
    )
    check(
        "All 372 balanced candidate objectives",
        max_candidate_score_error,
        1e-11,
    )
    check("All candidate clean distortions", max_distortion_error, 1e-12)
    check("All candidate finite-record noise gains", max_gain_error, 1e-13)
    check(
        "All candidate gain-constraint classifications",
        float(feasible_mismatches),
        0,
    )
    feasible = [row for row in candidate_checks if row["independent_feasible"]]
    minimum = min(row["independent_score"] for row in feasible)
    allowed = [
        row for row in feasible if row["independent_score"] <= minimum * 1.01
    ]
    selected = sorted(
        allowed,
        key=lambda row: (
            row["length"],
            row["independent_score"],
            row["half_width_hz"],
            ["rectangular", "hann", "hamming"].index(row["window"]),
        ),
    )[0]
    check(
        "Declared selection rule preserves the frozen candidate",
        float(selected["candidate_id"] != frozen["selected"]["candidate_id"]),
        0,
    )
    write_rows(metadata / "r2_candidate_recheck.csv", candidate_checks)

    validation_manifest = json.loads((r23 / "seed_manifest.json").read_text())
    published = {
        (row["pair_id"], int(row["rate_hz"])): row
        for row in read_rows(r23 / "paired_trials.csv")
    }
    validation_paths = {
        tuple(row["spawn_key"]) for row in validation_manifest["runs"]
    }
    check(
        "400 held-out paths disjoint from design",
        float(
            len(validation_paths) != 400
            or not design_paths.isdisjoint(validation_paths)
            or any(path[2] != 1 for path in validation_paths)
        ),
        0,
    )
    clean = np.convolve(x, h, mode="same")
    clean_high = resample(clean, len(reference))
    validation_checks = []
    max_validation_error = 0.0
    hash_mismatches = 0
    for run in validation_manifest["runs"]:
        seed = np.random.SeedSequence(
            run["root_seed"],
            spawn_key=tuple(run["spawn_key"]),
            pool_size=run["pool_size"],
        )
        rng = np.random.Generator(np.random.PCG64(seed))
        e = rng.normal(0.0, np.sqrt(run["variance"]), x.shape)
        contaminated = x + e
        for values, key in [
            (e, "noise_values_sha256"),
            (contaminated, "contaminated_values_sha256"),
        ]:
            digest = hashlib.sha256(
                np.asarray(values, dtype="<f8").tobytes()
            ).hexdigest()
            hash_mismatches += digest != run[key]
        filtered = np.convolve(contaminated, h, mode="same")
        for grid, target, before, after, clean_output, margin in [
            (16000, x, contaminated, filtered, clean, 256),
            (
                48000,
                reference,
                resample(contaminated, len(reference)),
                resample(filtered, len(reference)),
                clean_high,
                768,
            ),
        ]:
            saved = published[(run["run_id"], grid)]
            before_mse = float(np.mean((before - target) ** 2))
            after_error = after - target
            after_mse = float(np.mean(after_error**2))
            power = float(np.mean(target**2))
            edge_error = np.r_[after_error[:margin], after_error[-margin:]]
            end = len(target) - margin
            distortion = clean_output - target
            residual = after - clean_output
            values = {
                "no_filter_mse": before_mse,
                "filter_mse": after_mse,
                "no_filter_nmse": before_mse / power,
                "filter_nmse": after_mse / power,
                "no_filter_rmse": np.sqrt(before_mse),
                "filter_rmse": np.sqrt(after_mse),
                "no_filter_error_snr_db": 10 * np.log10(power / before_mse),
                "filter_error_snr_db": 10 * np.log10(power / after_mse),
                "paired_gain_db": 10 * np.log10(before_mse / after_mse),
                "paired_mse_reduction": before_mse - after_mse,
                "filter_edge_mse": np.mean(edge_error**2),
                "filter_interior_mse": np.mean(after_error[margin:end] ** 2),
                "clean_filter_mse": np.mean(distortion**2),
                "filtered_noise_mse": np.mean(residual**2),
                "cross_term": 2 * np.mean(distortion * residual),
            }
            error = max(
                abs(value - float(saved[key])) for key, value in values.items()
            )
            max_validation_error = max(max_validation_error, error)
            validation_checks.append(
                {
                    "run_id": run["run_id"],
                    "rate_hz": grid,
                    "independent_no_filter_mse": before_mse,
                    "independent_filter_mse": after_mse,
                    "maximum_metric_difference": error,
                }
            )
    check("All 400 Gaussian draws replay exactly", float(hash_mismatches), 0)
    check(
        "All 800 validation rows via independent direct/SciPy recovery",
        max_validation_error,
        1e-10,
    )
    write_rows(metadata / "r2_validation_recheck.csv", validation_checks)
    write_rows(metadata / "r2_strict_checks.csv", checks)
    report = {
        "status": "All independent Stage 2 numerical checks passed",
        "coefficient_sha256": coefficient_hash,
        "candidate_count": len(candidate_checks),
        "candidate_record_measurements": len(candidate_checks)
        * len(noise_rows),
        "feasible_candidates": len(feasible),
        "validation_records": len(validation_manifest["runs"]),
        "validation_grid_rows": len(validation_checks),
        "independent_candidate_fft_length": fft_length,
        "independent_kernel": (
            "Finite cosine-series sum evaluated with sinc; "
            "no production IFFT constructor"
        ),
        "validation_recovery": (
            "np.convolve(mode=same) and scipy.signal.resample; "
            "no production filter/reconstruction functions"
        ),
        "selection_audit_only": True,
        "filter_retuned": False,
        "new_random_trials": False,
        "checks": checks,
        "audit_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (metadata / "r2_strict_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Stage 2 strict audit: {len(checks)} checks; "
        f"{len(candidate_checks)} candidates; 400 validation records."
    )


if __name__ == "__main__":
    main()
