"""Design R3.1 using the quantisation example endpoint quantiser and measured mean squares."""

from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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
from stage2.r23.validation import array_hash, file_hash  # noqa: E402
from stage3.bitstream import bits_to_labels, labels_to_bits  # noqa: E402

Floats = NDArray[np.float64]
Indices = NDArray[np.uint32]


def dequantise(labels: Indices, bits: int, half_range: float) -> Floats:
    """Apply the reference endpoint inverse mapping in original amplitude units.

    Args:
        labels: Integer level indices from zero to 2**bits - 1.
        bits: Fixed width, one through sixteen.
        half_range: Positive fixed amplitude A.

    Returns:
        Float64 levels including both -A and +A.
    """
    values = np.asarray(labels)
    if (
        not isinstance(bits, (int, np.integer))
        or not 1 <= bits <= 16
        or not np.isfinite(half_range)
        or half_range <= 0
        or values.ndim != 1
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
        or np.any(values >= 2**bits)
    ):
        raise ValueError("Use valid integer labels and a fixed positive A.")
    return half_range * (
        np.asarray(labels, dtype=np.float64) / (2**bits - 1) * 2 - 1
    )


def quantise(
    samples: Floats, bits: int, half_range: float, epsilon: float = 1e-9
) -> tuple[Indices, Floats, Floats]:
    """Scale, round and invert as in quantisation example, with explicit saturation.

    Args:
        samples: Finite nonempty real one-dimensional quantiser input.
        bits: Fixed width, one through sixteen.
        half_range: Fixed positive A, not estimated inside this function.
        epsilon: reference shift in normalised amplitude units.

    Returns:
        Integer labels, quantised amplitudes and quantised-minus-input error.
    """
    values = np.asarray(samples)
    if (
        values.ndim != 1
        or not values.size
        or np.iscomplexobj(values)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Use a nonempty finite real one-dimensional input.")
    if (
        not isinstance(bits, (int, np.integer))
        or not 1 <= bits <= 16
        or not np.isfinite(half_range)
        or half_range <= 0
        or not np.isfinite(epsilon)
        or not 0 <= epsilon <= 1e-6
    ):
        raise ValueError("Invalid width, fixed half range or epsilon.")
    levels = 2**bits
    normalised = np.clip(values, -half_range, half_range) / half_range
    scaled = (normalised + 1.0 - epsilon) / 2.0 * (levels - 1)
    labels = np.clip(np.round(scaled), 0, levels - 1).astype(np.uint32)
    quantised = dequantise(labels, bits, half_range)
    return labels, quantised, quantised - values


def boundary_checks(results: Path, half_range: float) -> list[dict]:
    """Check actual reference arithmetic, range limits, ties and bit significance.

    Args:
        results: Existing output directory for transparent probe CSV.
        half_range: Fixed range used by the experiment.

    Returns:
        Named numerical checks, raising if a mathematical identity fails.
    """
    checks = []
    probes = []
    eps = 1e-9

    def check(name: str, error: float, tolerance: float) -> None:
        """Append one checked identity or raise on failure.

        Args:
            name: What is verified.
            error: Nonnegative numerical discrepancy.
            tolerance: Allowed floating point discrepancy.

        Returns:
            None. Append a passing row or raise.
        """
        if not np.isfinite(error) or error > tolerance:
            raise ArithmeticError(name)
        checks.append({"check": name, "error": error, "tolerance": tolerance})

    for bits in range(4, 17):
        levels = 2**bits
        values = np.linspace(-1, 1, 10001)
        lab_index = np.round((values + 1 - eps) / 2 * (levels - 1))
        lab_output = lab_index / (levels - 1) * 2 - 1
        _, actual, _ = quantise(values, bits, 1.0)
        check(
            f"B{bits}: actual quantisation example expression",
            float(np.max(np.abs(actual - lab_output))),
            0.0,
        )
        labels = np.arange(levels, dtype=np.uint32)
        decoded = bits_to_labels(labels_to_bits(labels, bits), bits)
        check(
            f"B{bits}: all labels MSB-first round trip",
            float(np.max(np.abs(decoded.astype(int) - labels))),
            0,
        )
        step = 2 * half_range / (levels - 1)
        test = np.array(
            [
                -half_range - step,
                -half_range,
                -half_range + step / 2,
                -half_range + step / 2 + half_range * eps / 2,
                0.0,
                half_range - step / 2,
                half_range,
                half_range + step,
            ]
        )
        indices, output, error = quantise(test, bits, half_range)
        exact, _, _ = quantise(test, bits, half_range, 0.0)
        granular = output - np.clip(test, -half_range, half_range)
        check(
            f"B{bits}: shifted half-step granular bound",
            max(
                0.0,
                float(np.max(np.abs(granular))) - step / 2 - half_range * eps,
            ),
            2e-13,
        )
        check(
            f"B{bits}: saturated endpoints",
            float(
                max(abs(output[0] + half_range), abs(output[-1] - half_range))
            ),
            0.0,
        )
        field = labels_to_bits(np.array([0], dtype=np.uint32), bits)
        field[0] = 1
        check(
            f"B{bits}: MSB weight",
            float(abs(int(bits_to_labels(field, bits)[0]) - 2 ** (bits - 1))),
            0,
        )
        for index in range(len(test)):
            probes.append(
                {
                    "bits": bits,
                    "input": test[index],
                    "index_lab_epsilon": int(indices[index]),
                    "index_zero_epsilon": int(exact[index]),
                    "quantised": output[index],
                    "error": error[index],
                    "overload": int(abs(test[index]) > half_range),
                    "step": step,
                }
            )
    write_csv(results / "boundary_probes.csv", probes)
    write_csv(results / "checks.csv", checks)
    return checks


def main() -> None:
    """Run the declared phase-0 study, choose B and freeze the quantiser.

    Returns:
        None. Save measurements, source/RNG manifests and the design freeze.
    """
    protocol = json.loads((ROOT / "design.json").read_text())
    registry_path = PACKAGE / "metadata/randomness_design.json"
    registry = json.loads(registry_path.read_text())
    if protocol["phase_id"] != 0:
        raise ValueError("R3.1 must use design phase 0.")
    kernel_path = PACKAGE / protocol["filter_file"]
    if file_hash(kernel_path) != protocol["filter_sha256"]:
        raise ValueError("The frozen R2 filter changed.")
    h = np.load(kernel_path)
    original = np.load(PACKAGE / "reference_signal.npy")
    x = np.load(PACKAGE / "stage1/r12/results/selected_samples.npy")
    if not np.array_equal(x, original[::3]):
        raise ValueError("R1 samples changed.")
    power = float(np.mean(x**2))
    clean = centred_filter(x, h)
    minimum_snr = min(c["target_snr_db"] for c in protocol["conditions"])
    sigma_max = math.sqrt(power * 10 ** (-minimum_snr / 10))
    peak = {
        "unfiltered": float(np.max(np.abs(x))),
        "filtered": float(np.max(np.abs(clean))),
    }
    gain = {"unfiltered": 1.0, "filtered": float(np.linalg.norm(h))}
    headroom = protocol["headroom_standard_deviations"]
    bounds = {
        key: peak[key] + headroom * sigma_max * gain[key] for key in peak
    }
    half_range = float(math.ceil(max(bounds.values())))
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    checks = boundary_checks(results, half_range)
    trials = []
    manifests = []
    example = None
    max_energy_residual = 0.0
    max_decomposition = 0.0
    max_bit_roundtrip = 0.0
    max_granular_excess = 0.0
    epsilon = protocol["normalised_epsilon"]
    alternating = (-1.0) ** np.arange(len(x))
    for condition in protocol["conditions"]:
        cid = condition["condition_id"]
        snr = condition["target_snr_db"]
        for repeat in range(protocol["repeats"]):
            path = (1, 3, 0, cid, repeat)
            rng = make_rng(path)
            initial = rng.bit_generator.state
            noisy, noise, variance = add_gaussian_noise(x, snr, rng)
            paired = {
                "filtered": centred_filter(noisy, h),
                "unfiltered": noisy,
            }
            manifests.append(
                {
                    "spawn_key": list(path),
                    "initial_state": initial,
                    "normal_call_shape": list(x.shape),
                    "target_snr_db": snr,
                    "realised_snr_db": float(
                        10 * np.log10(power / np.mean(noise**2))
                    ),
                    "model_variance": variance,
                    "noise_array_sha256": array_hash(noise),
                    "noisy_array_sha256": array_hash(noisy),
                    "filtered_array_sha256": array_hash(paired["filtered"]),
                    "pairing_group": f"r31-c{cid}-r{repeat}",
                }
            )
            for branch, z in paired.items():
                before_high = periodic_reconstruct(z, len(original))
                upstream = before_high - original
                baseline_mse = float(np.mean(upstream**2))
                clipped = np.clip(z, -half_range, half_range)
                clip_error = clipped - z
                for bits in protocol["bits_candidates"]:
                    labels, output, error = quantise(
                        z, bits, half_range, epsilon
                    )
                    recovered_labels = bits_to_labels(
                        labels_to_bits(labels, bits), bits
                    )
                    max_bit_roundtrip = max(
                        max_bit_roundtrip,
                        float(
                            np.max(
                                np.abs(labels.astype(int) - recovered_labels)
                            )
                        ),
                    )
                    reconstructed = periodic_reconstruct(output, len(original))
                    q_high = reconstructed - before_high
                    q_power = float(np.mean(error**2))
                    q_power_high = float(np.mean(q_high**2))
                    total_mse = float(np.mean((reconstructed - original) ** 2))
                    cross = float(2 * np.mean(upstream * q_high))
                    max_decomposition = max(
                        max_decomposition,
                        abs(total_mse - baseline_mse - q_power_high - cross),
                    )
                    nyquist = float(np.mean(error * alternating))
                    max_energy_residual = max(
                        max_energy_residual,
                        abs(q_power_high - q_power + 0.5 * nyquist**2),
                    )
                    step = 2 * half_range / (2**bits - 1)
                    granular = output - clipped
                    max_granular_excess = max(
                        max_granular_excess,
                        float(np.max(np.abs(granular)))
                        - step / 2
                        - half_range * epsilon,
                    )
                    zero_labels, _, zero_error = quantise(
                        z, bits, half_range, 0.0
                    )
                    pz = float(np.mean(z**2))
                    trials.append(
                        {
                            "condition_id": cid,
                            "target_snr_db": snr,
                            "repeat_id": repeat,
                            "branch": branch,
                            "bits": bits,
                            "half_range": half_range,
                            "step": step,
                            "input_power": pz,
                            "quantisation_mse_16000": q_power,
                            "error_mean": float(np.mean(error)),
                            "sqnr_db": float(10 * np.log10(pz / q_power)),
                            "model_sqnr_db": float(
                                10 * np.log10(12 * pz / step**2)
                            ),
                            "clip_count": int(np.count_nonzero(z != clipped)),
                            "clip_mse": float(np.mean(clip_error**2)),
                            "granular_mse": float(np.mean(granular**2)),
                            "clip_granular_cross": float(
                                2 * np.mean(clip_error * granular)
                            ),
                            "upstream_mse_48000": baseline_mse,
                            "quantisation_mse_48000": q_power_high,
                            "total_mse_48000": total_mse,
                            "upstream_quantisation_cross_48000": cross,
                            "component_ratio": q_power_high / baseline_mse,
                            "total_mse_change_fraction": (
                                total_mse / baseline_mse - 1
                            ),
                            "epsilon_changed_labels": int(
                                np.count_nonzero(labels != zero_labels)
                            ),
                            "zero_epsilon_quantisation_mse": float(
                                np.mean(zero_error**2)
                            ),
                            "used_levels": int(len(np.unique(labels))),
                            "payload_bits": len(x) * bits,
                            "payload_bits_per_second": 16000 * bits,
                        }
                    )
                if cid == 5 and repeat == 0 and branch == "filtered":
                    example = z.copy()
    write_csv(results / "trials.csv", trials)
    selection = []
    for bits in protocol["bits_candidates"]:
        rows = [r for r in trials if r["bits"] == bits]
        worst = max(rows, key=lambda r: r["component_ratio"])
        max_clip = max(r["clip_count"] for r in rows)
        selection.append(
            {
                "bits": bits,
                "worst_component_ratio": worst["component_ratio"],
                "limiting_condition_id": worst["condition_id"],
                "limiting_repeat_id": worst["repeat_id"],
                "limiting_branch": worst["branch"],
                "max_clip_count": max_clip,
                "passes": int(
                    worst["component_ratio"]
                    <= protocol["quantisation_component_budget"]
                    and max_clip == 0
                ),
                "payload_bits_per_second": 16000 * bits,
            }
        )
    passed = [r["bits"] for r in selection if r["passes"]]
    if not passed:
        raise ArithmeticError("No bit width satisfies the declared budget.")
    chosen = min(passed)
    write_csv(results / "bit_selection.csv", selection)
    sensitivity = []
    for budget in protocol["budget_sensitivity"]:
        acceptable = [
            r["bits"]
            for r in selection
            if r["worst_component_ratio"] <= budget
            and r["max_clip_count"] == 0
        ]
        sensitivity.append(
            {
                "component_budget": budget,
                "smallest_passing_bits": min(acceptable) if acceptable else "",
                "status": "pass" if acceptable else "no candidate",
            }
        )
    write_csv(results / "budget_sensitivity.csv", sensitivity)
    summary = []
    for condition in protocol["conditions"]:
        for branch in protocol["branches"]:
            for bits in protocol["bits_candidates"]:
                rows = [
                    r
                    for r in trials
                    if r["condition_id"] == condition["condition_id"]
                    and r["branch"] == branch
                    and r["bits"] == bits
                ]
                summary.append(
                    {
                        "condition_id": condition["condition_id"],
                        "target_snr_db": condition["target_snr_db"],
                        "branch": branch,
                        "bits": bits,
                        "repeats": len(rows),
                        **{
                            "mean_"
                            + key: float(np.mean([r[key] for r in rows]))
                            for key in [
                                "sqnr_db",
                                "model_sqnr_db",
                                "quantisation_mse_16000",
                                "upstream_mse_48000",
                                "total_mse_48000",
                                "component_ratio",
                                "total_mse_change_fraction",
                            ]
                        },
                        "min_sqnr_db": min(r["sqnr_db"] for r in rows),
                        "max_sqnr_db": max(r["sqnr_db"] for r in rows),
                        "max_component_ratio": max(
                            r["component_ratio"] for r in rows
                        ),
                        "clip_count": sum(r["clip_count"] for r in rows),
                    }
                )
    write_csv(results / "summary.csv", summary)
    if example is None:
        raise ArithmeticError("Missing predetermined example.")
    labels, output, error = quantise(example, chosen, half_range, epsilon)
    np.save(results / "example_labels.npy", labels)
    np.save(results / "example_quantised.npy", output)
    (results / "example_bits_msb.bin").write_bytes(
        np.packbits(labels_to_bits(labels, chosen), bitorder="big").tobytes()
    )
    np.savetxt(
        results / "example_waveform.csv",
        np.column_stack(
            (
                np.arange(len(x)),
                np.arange(len(x)) / 16000,
                example,
                labels,
                output,
                error,
            )
        ),
        delimiter=",",
        comments="",
        header="index,time_s,input,label,quantised,error",
        fmt=["%d", "%.17g", "%.17g", "%d", "%.17g", "%.17g"],
    )
    selected = [r for r in trials if r["bits"] == chosen]
    if max_bit_roundtrip or max_energy_residual > 1e-11:
        raise ArithmeticError("Bit or Fourier energy check failed.")
    if max_decomposition > 1e-11 or max_granular_excess > 1e-11:
        raise ArithmeticError("Error decomposition or granular bound failed.")
    source_paths = [
        ROOT / "quantisation.py",
        ROOT / "design.json",
        PACKAGE / "stage3/bitstream.py",
        PACKAGE / "degradation.py",
        PACKAGE / "stage1/r12/sampling_rate.py",
        PACKAGE / "stage2/r21/filtering_approach.py",
        PACKAGE / "stage2/r23/validation.py",
        PACKAGE / "reference_signal.npy",
        kernel_path,
        PACKAGE / "stage1/r12/results/selected_samples.npy",
    ]
    frozen = {
        "status": "R3.1 quantiser frozen from phase-0 design evidence",
        "bits": chosen,
        "half_range": half_range,
        "levels": 2**chosen,
        "step": 2 * half_range / (2**chosen - 1),
        "normalised_epsilon": epsilon,
        "label_type": "uint32",
        "bit_order": protocol["bit_order"],
        "range_rule": protocol["range_handling"],
        "component_budget": protocol["quantisation_component_budget"],
        "worst_observed_component_ratio": max(
            r["component_ratio"] for r in selected
        ),
        "maximum_observed_total_mse_change_fraction": max(
            r["total_mse_change_fraction"] for r in selected
        ),
        "source_sha256": {
            p.relative_to(PACKAGE).as_posix(): file_hash(p)
            for p in source_paths
        },
        "headroom": {
            "minimum_snr_db": minimum_snr,
            "sigma_max": sigma_max,
            "clean_peaks": peak,
            "pointwise_sd_gain_upper": gain,
            "six_sigma_amplitude_bounds": bounds,
            "per_sample_two_sided_six_sigma_tail": math.erfc(
                headroom / math.sqrt(2)
            ),
            "two_branch_record_union_bound": 2
            * len(x)
            * math.erfc(headroom / math.sqrt(2)),
            "guarantees_zero_overload": False,
        },
        "evidence_scope": (
            "40 Gaussian design records, 80 branches; "
            "no R3/R4 held-out coding validation"
        ),
        "no_bsc_or_parameter_leaf_0_1_used": True,
    }
    (results / "quantiser_freeze.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )
    manifest = {
        "root_seed": registry["root_seed"],
        "pool_size": 4,
        "bit_generator": "PCG64",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "phase_id": 0,
        "study_id": 3,
        "trial_paths": manifests,
        "source_sha256": frozen["source_sha256"],
    }
    (results / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    verification = {
        "status": "passed",
        "noise_records": len(manifests),
        "branch_records": len(manifests) * 2,
        "bit_depth_measurements": len(trials),
        "boundary_checks": len(checks),
        "all_labels_checked_count": sum(2**b for b in range(4, 17)),
        "maximum_bit_roundtrip_error": max_bit_roundtrip,
        "maximum_fourier_energy_residual": max_energy_residual,
        "maximum_mse_decomposition_residual": max_decomposition,
        "maximum_granular_bound_excess": max(0, max_granular_excess),
        "unique_noise_paths": len({tuple(m["spawn_key"]) for m in manifests})
        == len(manifests),
        "all_paths_phase_0_study_3": True,
        "selected_bits": chosen,
        "selected_half_range": half_range,
        "selected_overload_count": sum(r["clip_count"] for r in selected),
        "selected_epsilon_changed_labels": sum(
            r["epsilon_changed_labels"] for r in selected
        ),
        "coefficient_sha256": file_hash(kernel_path),
        "parameter_p_still_null": (
            registry["bit_flip_baseline_probability"] is None
        ),
    }
    (PACKAGE / "metadata/r31_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(
        f"R3.1: B={chosen}, A={half_range:g}; "
        f"{len(trials)} bit-depth measurements; checks passed."
    )


if __name__ == "__main__":
    main()
