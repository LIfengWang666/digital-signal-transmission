"""Validate the frozen R2.2 filter using new phase-1 Gaussian records.

The loops and mean-square measurements follow the Labs. Independent-record
confidence intervals are an explicitly declared statistical extension.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from degradation import add_gaussian_noise  # noqa: E402
from stage1.r12.sampling_rate import periodic_reconstruct  # noqa: E402
from stage2.r21.filtering_approach import (  # noqa: E402
    centred_filter,
    make_rng,
    write_csv,
)

FloatArray = NDArray[np.float64]


def file_hash(path: Path) -> str:
    """Hash a stored source or measurement.

    Args:
        path: Existing file.

    Returns:
        Hexadecimal SHA-256.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(values: FloatArray) -> str:
    """Hash actual contiguous float64 values, without an NPY header.

    Args:
        values: Real numerical record.

    Returns:
        SHA-256 of its C-order little-endian float64 bytes.
    """
    return hashlib.sha256(
        np.asarray(values, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def mean_interval(values: FloatArray) -> tuple[float, float, float, float]:
    """Summarise independent whole-record Monte Carlo measurements.

    Args:
        values: At least two finite scalar measurements, one per repeat.

    Returns:
        Mean, sample standard deviation, and pointwise approximate
        two-sided 95% confidence limits for the population mean.
    """
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Intervals require at least two finite records.")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    half_width = (
        float(t.ppf(0.975, len(values) - 1)) * sd / np.sqrt(len(values))
    )
    return mean, sd, mean - half_width, mean + half_width


def measure_pair(
    reference: FloatArray,
    untreated: FloatArray,
    filtered: FloatArray,
    clean_filtered: FloatArray,
    edge: int,
) -> dict[str, float]:
    """Measure both branches on the same full interval and amplitude scale.

    Args:
        reference: Unchanged clean values at this evaluation rate.
        untreated: No-filter recovery.
        filtered: Frozen-filter recovery.
        clean_filtered: The same recovery path with zero added noise.
        edge: Fixed number of diagnostic samples at each endpoint.

    Returns:
        Full-record fidelity, paired gains, boundary diagnostics and
        the deterministic/noise/cross-term error decomposition.
    """
    power = float(np.mean(reference**2))
    before_error = untreated - reference
    after_error = filtered - reference
    distortion = clean_filtered - reference
    residual = filtered - clean_filtered
    before = float(np.mean(before_error**2))
    after = float(np.mean(after_error**2))
    interior_end = len(reference) - edge
    edge_error = np.concatenate((after_error[:edge], after_error[-edge:]))
    clean_mse = float(np.mean(distortion**2))
    residual_mse = float(np.mean(residual**2))
    cross = float(2 * np.mean(distortion * residual))
    return {
        "no_filter_mse": before,
        "filter_mse": after,
        "no_filter_nmse": before / power,
        "filter_nmse": after / power,
        "no_filter_rmse": float(np.sqrt(before)),
        "filter_rmse": float(np.sqrt(after)),
        "no_filter_error_snr_db": float(10 * np.log10(power / before)),
        "filter_error_snr_db": float(10 * np.log10(power / after)),
        "paired_gain_db": float(10 * np.log10(before / after)),
        "paired_mse_reduction": before - after,
        "filter_edge_mse": float(np.mean(edge_error**2)),
        "filter_interior_mse": float(
            np.mean(after_error[edge:interior_end] ** 2)
        ),
        "clean_filter_mse": clean_mse,
        "filtered_noise_mse": residual_mse,
        "cross_term": cross,
        "decomposition_residual": after - clean_mse - residual_mse - cross,
    }


def save_example(
    path: Path,
    reference: FloatArray,
    untreated: FloatArray,
    filtered: FloatArray,
    rate: int,
) -> None:
    """Save a predetermined full record for independent plots and checks.

    Args:
        path: CSV destination.
        reference: Clean comparison values.
        untreated: No-filter recovery values.
        filtered: Filtered recovery values.
        rate: Samples per second on this grid.

    Returns:
        None. Write every sample, including both endpoints.
    """
    columns = np.column_stack(
        (
            np.arange(len(reference)),
            np.arange(len(reference)) / rate,
            reference,
            untreated,
            filtered,
            untreated - reference,
            filtered - reference,
        )
    )
    np.savetxt(
        path,
        columns,
        delimiter=",",
        header="index,time_s,reference,no_filter,filtered,"
        "no_filter_error,filter_error",
        comments="",
        fmt=["%d", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g"],
    )


def main() -> None:
    """Run the fixed validation protocol and save its measured evidence.

    Returns:
        None. Write CSV, a seed/source manifest and verification JSON.
    """
    design_path = ROOT / "design.json"
    protocol = json.loads(design_path.read_text(encoding="utf-8"))
    registry_path = PACKAGE / "metadata/randomness_design.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    kernel_path = PACKAGE / protocol["coefficient_file"]
    frozen_path = PACKAGE / "stage2/r22/results/filter_freeze.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    coefficient_hash = file_hash(kernel_path)
    if coefficient_hash != protocol["coefficient_sha256"]:
        raise ValueError("The predeclared coefficient hash has changed.")
    if coefficient_hash != frozen["coefficient_sha256"]:
        raise ValueError("The R2.2 freeze and stored coefficients disagree.")
    if protocol["phase_id"] != 1 or protocol["repeats"] != 50:
        raise ValueError("Do not silently change the validation protocol.")
    h = np.load(kernel_path)
    source = PACKAGE / "reference_signal.npy"
    sample_path = PACKAGE / "stage1/r12/results/selected_samples.npy"
    original = np.load(source)
    x = np.load(sample_path)
    if not np.array_equal(x, original[::3]):
        raise ValueError("The validation samples differ from R1.2.")
    n = len(x)
    edge = protocol["fixed_edge_samples_16000hz"]
    clean = centred_filter(x, h)
    clean_high = periodic_reconstruct(clean, len(original))
    clean_recovery = periodic_reconstruct(x, len(original))
    lags = np.arange(-(len(h) // 2), len(h) // 2 + 1)
    gain = float(np.sum(h**2 * (1 - np.abs(lags) / n)))
    alternating = (-1.0) ** np.arange(n)
    nyquist_weights = centred_filter(alternating, h) / n
    gain_high = gain - 0.5 * float(np.sum(nyquist_weights**2))
    prediction_base = {
        16000: (float(np.mean((clean - x) ** 2)), gain, 1.0),
        48000: (
            float(np.mean((clean_high - original) ** 2)),
            gain_high,
            1 - 1 / (2 * n),
        ),
    }
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    trials = []
    manifests = []
    example_hashes = {}
    max_parseval_residual = 0.0
    max_linear_residual = 0.0
    max_direct_residual = 0.0
    prediction_rows = []
    for condition in protocol["conditions"]:
        cid = condition["condition_id"]
        snr = condition["target_snr_db"]
        registered = [
            row
            for row in registry["condition_registry"]
            if row["study_id"] == 2
            and row["phase_id"] == 1
            and row["condition_id"] == cid
        ]
        if len(registered) != 1 or registered[0]["target_snr_db"] != snr:
            raise ValueError(
                "A condition is missing from the stable registry."
            )
        variance = float(np.mean(x**2)) * 10 ** (-snr / 10)
        for rate, (
            distortion,
            noise_gain,
            untreated_gain,
        ) in prediction_base.items():
            expected_before = variance * untreated_gain
            expected_after = distortion + variance * noise_gain
            prediction_rows.append(
                {
                    "condition_id": cid,
                    "target_snr_db": snr,
                    "rate_hz": rate,
                    "clean_mse": distortion,
                    "finite_noise_gain": noise_gain,
                    "no_filter_noise_gain": untreated_gain,
                    "expected_no_filter_mse": expected_before,
                    "expected_filter_mse": expected_after,
                    "gain_from_expected_mse_db": float(
                        10 * np.log10(expected_before / expected_after)
                    ),
                }
            )
        for repeat in range(protocol["repeats"]):
            path = (1, 2, 1, cid, repeat)
            rng = make_rng(path)
            initial_state = rng.bit_generator.state
            y, noise, actual_variance = add_gaussian_noise(x, snr, rng)
            if not np.isclose(actual_variance, variance, rtol=1e-14):
                raise ValueError("The declared noise model has changed.")
            filtered = centred_filter(y, h)
            untreated_high = periodic_reconstruct(y, len(original))
            filtered_high = periodic_reconstruct(filtered, len(original))
            realised = float(10 * np.log10(np.mean(x**2) / np.mean(noise**2)))
            pair_id = f"R2-phase1-c{cid}-r{repeat}"
            for rate, reference, untreated, output, clean_output in [
                (16000, x, y, filtered, clean),
                (48000, original, untreated_high, filtered_high, clean_high),
            ]:
                metrics = measure_pair(
                    reference,
                    untreated,
                    output,
                    clean_output,
                    edge * (rate // 16000),
                )
                trials.append(
                    {
                        "pair_id": pair_id,
                        "condition_id": cid,
                        "repeat_id": repeat,
                        "target_snr_db": snr,
                        "realised_sample_snr_db": realised,
                        "rate_hz": rate,
                        **metrics,
                    }
                )
            v = filtered - x
            v_high = periodic_reconstruct(v, len(original))
            predicted_energy = (
                np.mean(v**2) - 0.5 * np.mean(alternating * v) ** 2
            )
            max_parseval_residual = max(
                max_parseval_residual,
                abs(float(np.mean(v_high**2) - predicted_energy)),
            )
            if repeat == 0:
                direct = np.convolve(y, h, mode="full")
                delay = (len(h) - 1) // 2
                end = delay + n
                max_direct_residual = max(
                    max_direct_residual,
                    float(np.max(np.abs(direct[delay:end] - filtered))),
                )
                max_linear_residual = max(
                    max_linear_residual,
                    float(
                        np.max(
                            np.abs(filtered - clean - centred_filter(noise, h))
                        )
                    ),
                )
            if (
                cid in protocol["example_condition_ids"]
                and repeat == protocol["example_repeat_id"]
            ):
                for rate, reference, untreated, output in [
                    (16000, x, y, filtered),
                    (48000, original, untreated_high, filtered_high),
                ]:
                    name = f"example_c{cid}_r{repeat}_{rate}hz.csv"
                    save_example(
                        results / name, reference, untreated, output, rate
                    )
                    example_hashes[name] = file_hash(results / name)
                replay, replay_noise, _ = add_gaussian_noise(
                    x, snr, make_rng(path)
                )
                if not (
                    np.array_equal(y, replay)
                    and np.array_equal(noise, replay_noise)
                ):
                    raise ValueError("Exact replay of an example has failed.")
            manifests.append(
                {
                    "run_id": pair_id,
                    "root_seed": registry["root_seed"],
                    "pool_size": registry["pool_size"],
                    "bit_generator": "PCG64",
                    "spawn_key": path,
                    "study_id": 2,
                    "phase_id": 1,
                    "condition_id": cid,
                    "repeat_id": repeat,
                    "initial_state": initial_state,
                    "target_snr_db": snr,
                    "realised_snr_db": realised,
                    "variance": actual_variance,
                    "call": "normal(0.0, sqrt(variance), (32000,))",
                    "dtype": "float64",
                    "noise_values_sha256": array_hash(noise),
                    "contaminated_values_sha256": array_hash(y),
                    "filtered_values_sha256": array_hash(filtered),
                    "unfiltered_reconstruction_values_sha256": array_hash(
                        untreated_high
                    ),
                    "filtered_reconstruction_values_sha256": array_hash(
                        filtered_high
                    ),
                    "pairing_group": pair_id,
                }
            )
    write_csv(results / "paired_trials.csv", trials)
    write_csv(results / "theory.csv", prediction_rows)
    summaries = []
    # The first six CSV fields identify a run and its evaluation grid.
    metric_names = list(trials[0])[6:]
    for condition in protocol["conditions"]:
        cid = condition["condition_id"]
        for rate in [16000, 48000]:
            rows = [
                row
                for row in trials
                if row["condition_id"] == cid and row["rate_hz"] == rate
            ]
            for metric in metric_names:
                values = np.array([row[metric] for row in rows])
                mean, sd, low, high = mean_interval(values)
                summaries.append(
                    {
                        "condition_id": cid,
                        "target_snr_db": condition["target_snr_db"],
                        "rate_hz": rate,
                        "metric": metric,
                        "repeats": len(rows),
                        "mean": mean,
                        "sample_sd": sd,
                        "ci95_low": low,
                        "ci95_high": high,
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                        "positive_count": int(np.count_nonzero(values > 0)),
                    }
                )
    write_csv(results / "summary.csv", summaries)
    source_paths = [
        source,
        sample_path,
        kernel_path,
        frozen_path,
        design_path,
        Path(__file__),
        PACKAGE / "degradation.py",
        PACKAGE / "stage1/r12/sampling_rate.py",
        PACKAGE / "stage2/r21/filtering_approach.py",
    ]
    manifest = {
        "protocol": "../design.json",
        "protocol_sha256": file_hash(design_path),
        "root_seed": registry["root_seed"],
        "pool_size": registry["pool_size"],
        "bit_generator": "PCG64",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "parameter_range_and_mapping": {
            "snr_baseline_range_db": registry["snr_db_range"],
            "snr_mapping": registry["snr_mapping"],
            "snr_baseline_db": registry["snr_baseline_db"],
            "other_levels": "Fixed diagnostic levels; not random SNR draws",
            "bit_flip_probability": None,
        },
        "input_and_code_sha256": {
            str(path.relative_to(PACKAGE)).replace("\\", "/"): file_hash(path)
            for path in source_paths
        },
        "saved_example_sha256": example_hashes,
        "trial_csv_sha256": file_hash(results / "paired_trials.csv"),
        "summary_csv_sha256": file_hash(results / "summary.csv"),
        "theory_csv_sha256": file_hash(results / "theory.csv"),
        "runs": manifests,
    }
    (results / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    decomposition_residual = max(
        abs(row["decomposition_residual"]) for row in trials
    )
    paths = {tuple(row["spawn_key"]) for row in manifests}
    old_paths = {tuple(path) for path in frozen["source_spawn_keys"]}
    checks = {
        "original_sample_identity": np.array_equal(x, original[::3]),
        "frozen_coefficients_unchanged": file_hash(kernel_path)
        == coefficient_hash,
        "400_unique_validation_paths": len(paths) == 400,
        "disjoint_from_design_paths": paths.isdisjoint(old_paths),
        "eight_conditions_two_full_grids": len(trials) == 800,
        "clean_reference_reconstruction": np.max(
            np.abs(clean_recovery - original)
        )
        < 1e-10,
        "direct_vs_fft_eight_first_repeats": max_direct_residual < 1e-11,
        "filter_linearity_eight_first_repeats": max_linear_residual < 1e-11,
        "parseval_nyquist_identity_all_records": max_parseval_residual < 1e-11,
        "error_decomposition_all_records": decomposition_residual < 1e-11,
        "finite_gain_matches_freeze": abs(
            gain - frozen["selected"]["noise_gain_full"]
        )
        < 1e-14,
        "clean_distortion_matches_freeze": abs(
            prediction_base[16000][0] - frozen["selected"]["clean_full_mse"]
        )
        < 1e-14,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    verification = {
        "status": "R2.3 validation checks passed",
        "checks": {key: bool(value) for key, value in checks.items()},
        "coefficient_sha256": coefficient_hash,
        "records": len(manifests),
        "grid_measurements": len(trials),
        "max_direct_fft_error": max_direct_residual,
        "max_linearity_error": max_linear_residual,
        "max_parseval_energy_error": max_parseval_residual,
        "max_decomposition_error": decomposition_residual,
        "prediction_base": prediction_base,
        "sample_crossover_db": float(
            10
            * np.log10(np.mean(x**2) * (1 - gain) / prediction_base[16000][0])
        ),
        "phase_1_used_for_parameter_selection": False,
        "confidence_intervals": protocol["interval_method"],
    }
    (PACKAGE / "metadata/r23_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print("R2.3: 400 paired validation records; 800 full-grid measurements.")
    print("R2.3: all 12 numerical/provenance checks passed.")


if __name__ == "__main__":
    main()
