"""Assemble the four fixed paths with a truth-free receiver."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from stage1.r12.sampling_rate import periodic_reconstruct
from stage2.r21.filtering_approach import centred_filter
from stage3.bitstream import Bits, bits_to_labels, labels_to_bits
from stage3.r31.quantisation import dequantise, quantise
from stage3.r32.codes import decode, encode
from stage3.r32.pilot_comparison import bits_hash
from stage4.r41.metrics import FloatArray, reconstruction_metrics

ROOT = Path(__file__).resolve().parents[2]


def load_system() -> dict:
    """Load the frozen samples, filter, codebook and common-grid reference.

    Returns:
        Public parameters, transmitter arrays and evaluation-only baselines.
    """
    design = json.loads(
        (ROOT / "stage4/r42/design.json").read_text(encoding="utf-8")
    )
    reference = np.load(ROOT / "reference_signal.npy", allow_pickle=False)
    x = np.load(ROOT / "stage1/r12/results/selected_samples.npy")
    h = np.load(ROOT / "stage2/r22/results/selected_coefficients.npy")
    if reference.shape != (96000,) or not np.array_equal(reference[::3], x):
        raise ValueError(
            "Use the original reference and frozen 16 kHz samples."
        )
    clean = {0: x, 1: centred_filter(x, h)}
    clean_high = {
        f: periodic_reconstruct(signal, len(reference))
        for f, signal in clean.items()
    }
    return {
        "reference": reference,
        "x": x,
        "h": h,
        "bits": design["bits"],
        "half_range": design["half_range"],
        "epsilon": design["epsilon"],
        "clean": clean,
        "clean_high": clean_high,
        "sampling_error": clean_high[0] - reference,
        "nyquist_sign": np.where(np.arange(len(x)) % 2 == 0, 1.0, -1.0),
    }


def prepare_bank(
    system: dict, noisy: FloatArray, noise: FloatArray
) -> dict[int, dict]:
    """Prepare both filter branches once for a common Gaussian input.

    Args:
        system: Frozen system and evaluation baselines from load_system.
        noisy: The exact samples returned by the Gaussian interface.
        noise: Its returned measurement-only noise, not a receiver input.

    Returns:
        Two banks with reference labels, payloads, encoded streams and diagnostics.
    """
    x, reference = system["x"], system["reference"]
    if noisy.shape != x.shape or not np.array_equal(noisy, x + noise):
        raise ValueError("The paired input must be the same Gaussian record.")
    banks = {}
    for f in [0, 1]:
        z = centred_filter(noisy, system["h"]) if f else noisy.copy()
        filtered_noise = centred_filter(noise, system["h"]) if f else noise
        labels, levels, _ = quantise(
            z, system["bits"], system["half_range"], system["epsilon"]
        )
        payload = labels_to_bits(labels, system["bits"])
        coded, padding = encode(payload, "hamming74")
        if padding != 0 or len(payload) != 448000 or len(coded) != 784000:
            raise ValueError("Frozen R4 framing must have zero padding.")
        before = periodic_reconstruct(levels, len(reference))
        baseline = periodic_reconstruct(z, len(reference))
        banks[f] = {
            "z": z,
            "levels": levels,
            "payload": payload,
            "sent": {"uncoded": payload, "hamming74": coded},
            "before": before,
            "overload_samples": int(
                np.count_nonzero(np.abs(z) > system["half_range"])
            ),
            "errors": {
                "sampling": system["sampling_error"],
                "filter": system["clean_high"][f] - system["clean_high"][0],
                "gaussian": periodic_reconstruct(
                    filtered_noise, len(reference)
                ),
                "quantisation": before - baseline,
            },
            "low_errors": {
                "filter": system["clean"][f] - x,
                "gaussian": filtered_noise,
                "quantisation": levels - z,
            },
        }
    return banks


def receive_stream(
    received: Bits,
    code: str,
    payload_bits: int,
    bits: int,
    half_range: float,
    output_samples: int,
) -> dict[str, np.ndarray]:
    """Recover samples using only received bits and public format metadata.

    Args:
        received: Received BSC stream, never an error mask or true payload.
        code: uncoded or the frozen fixed hamming74 code.
        payload_bits: Public real information length, excluding ECC padding.
        bits: Public quantiser label width.
        half_range: Public fixed quantiser half range.
        output_samples: Public reconstructed reference-grid length.

    Returns:
        Decoded real bits, labels, 16 kHz levels and 48 kHz reconstruction.
    """
    if (
        code not in {"uncoded", "hamming74"}
        or payload_bits <= 0
        or not 1 <= bits <= 16
    ):
        raise ValueError("Use one of the two frozen layouts.")
    expected = (
        payload_bits if code == "uncoded" else 7 * ((payload_bits + 3) // 4)
    )
    if len(received) != expected or payload_bits % bits or output_samples < 1:
        raise ValueError("Received stream does not match public framing.")
    decoded, _ = decode(received, code)
    payload = decoded[:payload_bits]
    labels = bits_to_labels(payload, bits)
    levels = dequantise(labels, bits, half_range)
    high = periodic_reconstruct(levels, output_samples)
    return {
        "payload": payload,
        "labels": labels,
        "levels": levels,
        "reconstruction": high,
    }


def measure_path(
    system: dict,
    bank: dict,
    code: str,
    received: Bits,
    mask: np.ndarray,
    recovered: dict,
) -> tuple[dict, dict]:
    """Measure fidelity and error diagnostics outside the receiver.

    Args:
        system: Transmitter and reference evaluation context.
        bank: Prepared true payload and component errors for this branch.
        code: Public layout name.
        received: Received stream for a provenance hash.
        mask: Forward-channel measurement-only mask.
        recovered: Output of receive_stream, already decoded without truth.

    Returns:
        Metric/count/hash row and five-power/ten-cross-term decomposition.
    """
    high = recovered["reconstruction"]
    wrong = recovered["payload"] != bank["payload"]
    n, k, radius = (1, 1, 0) if code == "uncoded" else (7, 4, 1)
    failures = int(np.count_nonzero(np.any(wrong.reshape(-1, k), axis=1)))
    weights = np.sum(mask.reshape(-1, n), axis=1)
    if failures != int(np.count_nonzero(weights > radius)):
        raise ValueError("Block failures disagree with the perfect-code rule.")
    row = {
        **reconstruction_metrics(system["reference"], high),
        "payload_bits": len(wrong),
        "transmitted_bits": len(mask),
        "padding_bits": 0,
        "logical_rate_bits_s": len(mask) / 2,
        "channel_flips": int(np.count_nonzero(mask)),
        "decoded_wrong_bits": int(np.count_nonzero(wrong)),
        "block_failures": failures,
        "wrong_msb_bits": int(
            np.count_nonzero(wrong.reshape(-1, system["bits"])[:, 0])
        ),
        "wrong_samples": int(
            np.count_nonzero(np.any(wrong.reshape(-1, system["bits"]), axis=1))
        ),
        "overload_samples": bank["overload_samples"],
        "prechannel_nmse": reconstruction_metrics(
            system["reference"], bank["before"]
        )["nmse"],
        "transmitted_sha256": bits_hash(bank["sent"][code]),
        "received_sha256": bits_hash(received),
        "decoded_payload_sha256": bits_hash(recovered["payload"]),
        "wrong_pattern_sha256": bits_hash(wrong.astype(np.uint8)),
        "mask_sha256": bits_hash(mask.astype(np.uint8)),
    }
    errors = dict(bank["errors"])
    errors["channel"] = high - bank["before"]
    decomposition = {
        f"power_{name}": float(np.mean(error**2))
        for name, error in errors.items()
    }
    decomposition.update(
        {
            f"cross_{a}_{b}": float(2 * np.mean(errors[a] * errors[b]))
            for a, b in combinations(errors, 2)
        }
    )
    decomposition["mse_from_components"] = sum(decomposition.values())
    decomposition["mse_residual"] = abs(
        row["mse"] - decomposition["mse_from_components"]
    )
    decomposition["waveform_identity_residual"] = float(
        np.max(np.abs(high - system["reference"] - sum(errors.values())))
    )
    low = dict(bank["low_errors"])
    low["channel"] = recovered["levels"] - bank["levels"]
    residuals = []
    for name, values in low.items():
        predicted = (
            float(np.mean(values**2))
            - 0.5 * float(np.mean(values * system["nyquist_sign"])) ** 2
        )
        residuals.append(abs(float(np.mean(errors[name] ** 2)) - predicted))
    decomposition["nyquist_energy_residual"] = max(residuals)
    return row, decomposition


def paired_effects(values: dict[str, float]) -> dict[str, float]:
    """Calculate conditional switch effects on the declared NMSE scale.

    Args:
        values: One matched value for each of 00, 10, 01 and 11.

    Returns:
        Four error reductions and a difference-of-differences interaction.
    """
    a, b, c, d = [values[key] for key in ["00", "10", "01", "11"]]
    return {
        "filter_without_code": a - b,
        "filter_with_code": c - d,
        "code_without_filter": a - c,
        "code_with_filter": b - d,
        "interaction": a - b - c + d,
    }
