"""Post-review diagnostic: change packing at fixed 14 bit and reference Hamming74."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample

from stage1.r12.sampling_rate import periodic_reconstruct
from stage2.r21.filtering_approach import write_csv
from stage3.bitstream import bits_to_labels, labels_to_bits
from stage3.r31.quantisation import dequantise, quantise
from stage3.r32.codes import binary_words, decode, encode
from stage4.r42.pipeline import load_system

plt.switch_backend("Agg")
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent / "packing_diagnostic"


def moments(probability: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate all 128 BSC patterns through the fixed syndrome decoder.

    Args:
        probability: Independent transmitted-bit flip probability.

    Returns:
        Four residual-bit means, their covariance and pattern weights.
    """
    patterns = binary_words(7)
    errors, _ = decode(patterns.reshape(-1), "hamming74")
    errors = errors.reshape(-1, 4).astype(float)
    weight = patterns.sum(axis=1)
    weights = probability**weight * (1 - probability) ** (7 - weight)
    mean = weights @ errors
    covariance = (errors.T * weights) @ errors - np.outer(mean, mean)
    assert abs(weights.sum() - 1) < 1e-14
    return mean, covariance, weights


def layout_indices(count: int, name: str) -> np.ndarray:
    """Map each encoder input position to its natural sample-major bit.

    Args:
        count: Number of 14-bit labels, divisible by four.
        name: Sample-major baseline or MSB-first bit-plane alternative.

    Returns:
        A public invertible permutation, grouped into four-bit messages.
    """
    natural = np.arange(count * 14).reshape(count, 14)
    ordered = natural if name == "sample" else natural.T
    return ordered.reshape(-1, 4)


def expected_nmse(
    labels: np.ndarray, reference: np.ndarray, probability: float, name: str
) -> dict:
    """Evaluate exact BSC-expected full-grid NMSE for fixed quantised input.

    Args:
        labels: Fixed endpoint labels; no Gaussian input is drawn.
        reference: Measurement-only high-rate target.
        probability: BSC parameter.
        name: Public bit permutation before the unchanged Hamming encoder.

    Returns:
        Bias and variance contributions with the split-Nyquist correction.
        Exact here denotes the finite-pattern probability sum, subject to
        floating point evaluation, not an empirical confidence statement.
    """
    count = len(labels)
    indices = layout_indices(count, name)
    sample = indices // 14
    bit = indices % 14
    values = labels_to_bits(labels, 14).astype(float)[indices]
    slopes = 36 / (2**14 - 1) * 2.0 ** (13 - bit) * (1 - 2 * values)
    mean, covariance, _ = moments(probability)
    bias = np.zeros(count)
    np.add.at(bias, sample.ravel(), (slopes * mean).ravel())
    same_sample = sample[:, :, None] == sample[:, None, :]
    products = slopes[:, :, None] * slopes[:, None, :]
    trace = np.sum(products * covariance * same_sample)
    signs = np.where(sample % 2, -1.0, 1.0)
    nyquist_variance = np.sum(
        products * covariance * signs[:, :, None] * signs[:, None, :]
    )
    levels = dequantise(labels, 14, 18)
    mean_error = periodic_reconstruct(levels + bias, len(reference))
    bias_mse = float(np.mean((mean_error - reference) ** 2))
    variance = float(trace / count - nyquist_variance / (2 * count**2))
    power = float(np.mean(reference**2))
    return {
        "layout": name,
        "probability": probability,
        "expected_nmse": (bias_mse + variance) / power,
        "bias_mse": bias_mse,
        "variance_mse": variance,
        "expected_payload_BER": float(mean.mean()),
        "logical_rate_bits_s": 392000,
    }


def check_small_record() -> float:
    """Check the moment identity by explicit block outcomes and SciPy FFT.

    Returns:
        Maximum full-grid expected-MSE disagreement across both layouts.
        This uses all 128 outcomes of each independent block for N=8,
        rather than the covariance/Parseval calculation under test.
    """
    labels = np.array([19, 8140, 12340, 200, 1230, 9800, 4311, 16320])
    reference = resample(dequantise(labels, 14, 18), 24) + 0.03
    probability = 0.1
    errors, _ = decode(binary_words(7).ravel(), "hamming74")
    errors = errors.reshape(-1, 4).astype(float)
    _, _, weights = moments(probability)
    maximum = 0.0
    for name in ["sample", "plane"]:
        total_bias = np.zeros(24)
        total_variance = 0.0
        bits = labels_to_bits(labels, 14).astype(float)
        for indices in layout_indices(8, name):
            sample, bit = indices // 14, indices % 14
            slopes = 36 / (2**14 - 1) * 2.0 ** (13 - bit)
            slopes *= 1 - 2 * bits[indices]
            changes = np.zeros((128, 8))
            for j in range(4):
                changes[:, sample[j]] += errors[:, j] * slopes[j]
            high = resample(changes, 24, axis=1)
            mean_high = weights @ high
            total_bias += mean_high
            total_variance += float(
                weights @ np.mean(high**2, axis=1) - np.mean(mean_high**2)
            )
        before = resample(dequantise(labels, 14, 18), 24)
        expected = np.mean((before + total_bias - reference) ** 2)
        expected += total_variance
        result = expected_nmse(labels, reference, probability, name)
        maximum = max(
            maximum,
            abs(expected - result["expected_nmse"] * np.mean(reference**2)),
        )
    assert maximum < 1e-10
    return float(maximum)


def main() -> None:
    """Freeze, evaluate and report one supplementary mechanism comparison.

    Returns:
        None. Preserve all original R1--R5 data and confidence families.
    """
    HERE.mkdir(exist_ok=True)
    (ROOT / "report/images").mkdir(parents=True, exist_ok=True)
    (ROOT / "report/generated").mkdir(parents=True, exist_ok=True)
    system = load_system()
    mapping = json.loads(
        (ROOT / "metadata/randomness_design.json").read_text()
    )
    personal = float(mapping["bit_flip_baseline_probability"])
    probabilities = [0.0, personal, 0.01, 0.1]
    protocol = {
        "status": "Protocol written before this diagnostic is evaluated",
        "scope": (
            "Post-review mechanism diagnostic; outside R5 "
            "19-statistic family"
        ),
        "input": (
            "Original clean reference decimated to16kHz and frozen FIR; "
            "Gaussian absent"
        ),
        "fixed": (
            "14bit, A18, reference epsilon, Hamming74,784000sent bits,392kbit/s"
        ),
        "changed": "Sample-major versus MSB-first bit-plane permutation only",
        "probabilities": probabilities,
        "method": (
            "Exact block probability moments and full48kHz Parseval "
            "identity"
        ),
        "illustration_only": {
            "p": 0.01,
            "root_seed": 2470964,
            "spawn_key": [2, 51, 0, 0, 0, 0],
        },
        "no_optional_stopping_or_new_main_cohort_claim": True,
    }
    (HERE / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    labels, _, _ = quantise(system["clean"][1], 14, 18, 1e-9)
    rows = [
        expected_nmse(labels, system["reference"], p, name)
        for p in probabilities
        for name in ["sample", "plane"]
    ]
    write_csv(HERE / "expected_results.csv", rows)
    discrepancy = check_small_record()
    patterns = binary_words(7)
    residual, _ = decode(patterns.ravel(), "hamming74")
    messages = binary_words(4)
    for message in messages:
        word, _ = encode(message, "hamming74")
        actual, _ = decode((patterns ^ word).ravel(), "hamming74")
        assert np.array_equal(
            actual.reshape(-1, 4) ^ message, residual.reshape(-1, 4)
        )
    rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(2470964, spawn_key=(2, 51, 0, 0, 0, 0))
        )
    )
    mask = (rng.random(784000) < 0.01).astype(np.uint8)
    natural = labels_to_bits(labels, 14)
    waveform_rows = []
    errors = []
    for name in ["sample", "plane"]:
        indices = layout_indices(len(labels), name).ravel()
        sent, padding = encode(natural[indices], "hamming74")
        assert padding == 0 and len(sent) == len(mask)
        decoded, _ = decode(sent ^ mask, "hamming74")
        restored = np.empty_like(decoded)
        restored[indices] = decoded
        levels = dequantise(bits_to_labels(restored, 14), 14, 18)
        high = periodic_reconstruct(levels, 96000)
        error = high - system["reference"]
        errors.append(error)
        waveform_rows.append(
            {
                "layout": name,
                "single_record_nmse": float(
                    np.mean(error**2) / np.mean(system["reference"] ** 2)
                ),
                "wrong_payload_bits": int(
                    np.count_nonzero(restored != natural)
                ),
            }
        )
    write_csv(HERE / "illustration_metrics.csv", waveform_rows)
    plt.rcParams.update({"font.size": 11, "svg.hashsalt": "dst-packing"})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharex=True, sharey=True)
    time = np.arange(96000) / 48000
    for ax, error, label in zip(axes, errors, ["Sample-major", "Bit-plane"]):
        ax.plot(time, error, linewidth=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_title(label, loc="left", fontsize=11)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Waveform error")
    fig.suptitle(
        "Fixed 14 bit / Hamming(7,4); same mask at p=0.01",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(
        HERE / "waveform_errors.png",
        dpi=180,
        metadata={"Software": "DST"},
    )
    plt.close(fig)
    shutil_target = ROOT / "report/images/r51_packing_diagnostic.png"
    shutil_target.write_bytes((HERE / "waveform_errors.png").read_bytes())
    table = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"$p$ & Sample-major & Bit-plane \\",
        r"\midrule",
    ]
    for i in range(0, len(rows), 2):
        a, b = rows[i], rows[i + 1]
        table.append(
            f"{a['probability']:.6g} & {a['expected_nmse']:.6g} & "
            f"{b['expected_nmse']:.6g}" + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "report/generated/r51_packing_table.tex").write_text(
        "\n".join(table) + "\n"
    )
    check = {
        "status": "passed",
        "decoder_invariance_pairs_checked": 16 * 128,
        "independent_small_record_MSE_disagreement": discrepancy,
        "same_payload_quantiser_code_and_rate": True,
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "new_independent_statistical_observations_claimed": 0,
        "illustrative_channel_masks_generated": 1,
    }
    (HERE / "checks.json").write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    print("Fixed14bit packing diagnostic and independent moment check passed.")


if __name__ == "__main__":
    main()
