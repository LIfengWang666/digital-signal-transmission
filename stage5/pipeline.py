"""Apply single-decision alternatives using the same implemented operations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stage1.r12.sampling_rate import periodic_reconstruct
from stage2.r21.filtering_approach import centred_filter, write_csv
from stage2.r22.filter_parameters import frequency_kernel
from stage2.r23.validation import array_hash
from stage3.bitstream import bits_to_labels, labels_to_bits
from stage3.r31.quantisation import dequantise, quantise
from stage3.r32.codes import decode, encode, matrices
from stage3.r32.pilot_comparison import bits_hash
from stage4.r41.metrics import reconstruction_metrics
from stage4.r42.pipeline import load_system

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def load_design() -> dict:
    """Read the counterfactual conditions declared before new random draws.

    Returns:
        Fixed candidates, original constraints and prespecified R5.2 subset.
    """
    return json.loads((HERE / "design.json").read_text())


def build_system(save: bool = False) -> dict:
    """Build only the declared one-parameter FIR alternatives and layouts.

    Args:
        save: Write deterministic kernels, feasibility and coefficient CSV.

    Returns:
        Original reference/samples and the declared variants with their
        exact framing, gain feasibility and resource costs.
    """
    design = load_design()
    system = load_system()
    kernels, guard, coefficient_rows = {}, {}, []
    for choice in design["kernels"]:
        name, length = choice["id"], choice["taps"]
        h, _, _ = frequency_kernel(
            np.array(design["centres_hz"], dtype=float),
            choice["half_width_hz"],
            length,
            "hann",
            16000,
            32000,
        )
        if name == "base":
            assert np.array_equal(h, system["h"])
        lags = np.arange(length) - (length - 1) // 2
        frequencies = (
            np.array(design["centres_hz"])[:, None]
            + np.array(design["guard_offsets_hz"])
        ).reshape(-1)
        response = np.sum(
            np.exp(-2j * np.pi * frequencies[:, None] * lags / 16000) * h,
            axis=1,
        )
        deviation = float(np.max(np.abs(np.abs(response) - 1)))
        guard[name] = {
            "gain_max_deviation": deviation,
            "gain_guard_feasible": bool(
                deviation <= 0.01
                and np.all(response.real > 0)
                and np.max(np.abs(response.imag)) < 1e-12
            ),
        }
        kernels[name] = h
        coefficient_rows.extend(
            {"filter_id": name, "tap_index": i, "coefficient": float(v)}
            for i, v in enumerate(h)
        )
        if save:
            np.save(HERE / "results" / f"kernel_{name}.npy", h)
    variants = []
    for variant in design["variants"]:
        g, _ = matrices(variant["code"])
        k, n = g.shape
        payload = 32000 * variant["bits"]
        padding, sent = (-payload) % k, n * ((payload + k - 1) // k)
        h = kernels[variant["filter_id"]]
        expansion = sent / payload
        variants.append(
            {
                **variant,
                **guard[variant["filter_id"]],
                "payload_bits": payload,
                "padding_bits": padding,
                "transmitted_bits": sent,
                "logical_rate_bits_s": sent / 2,
                "n": n,
                "k": k,
                "correction_radius": (
                    2
                    if variant["code"] == "repetition51"
                    else 0 if variant["code"] == "uncoded" else 1
                ),
                "filter_taps": len(h),
                "group_delay_ms": (len(h) - 1) / (2 * 16000) * 1000,
                "expansion": expansion,
                "expansion_feasible": expansion <= 2,
                "original_constraints_feasible": guard[variant["filter_id"]][
                    "gain_guard_feasible"
                ]
                and expansion <= 2,
                "rate320_feasible": sent / 2 <= 320000,
            }
        )
    if save:
        write_csv(HERE / "results/variants.csv", variants)
        write_csv(HERE / "results/kernel_coefficients.csv", coefficient_rows)
    return {
        **system,
        "design": design,
        "kernels": kernels,
        "variants": variants,
    }


def prepare_inputs(system: dict, noisy: np.ndarray) -> dict:
    """Prepare unique filter/quantiser inputs once per Gaussian record.

    Args:
        system: Fixed reference, kernels and public variant definitions.
        noisy: One full32000-sample Gaussian-corrupted source record.

    Returns:
        Cached label, payload and encoded-stream banks by variant ID.
        Same filter/bit-depth inputs are not counted as new exposures.
    """
    quantisers, streams, banks = {}, {}, {}
    for variant in system["variants"]:
        key = variant["filter_id"], variant["bits"]
        if key not in quantisers:
            z = centred_filter(noisy, system["kernels"][key[0]])
            labels, levels, _ = quantise(z, key[1], 18, 1e-9)
            payload = labels_to_bits(labels, key[1])
            before = periodic_reconstruct(levels, 96000)
            quantisers[key] = {
                "z": z,
                "levels": levels,
                "payload": payload,
                "before": before,
                "overload_samples": int(np.count_nonzero(np.abs(z) > 18)),
                "prechannel_mse": float(
                    np.mean((before - system["reference"]) ** 2)
                ),
                "quantiser_input_sha256": array_hash(z),
                "quantisation_mse_16k": float(np.mean((levels - z) ** 2)),
            }
        q = quantisers[key]
        stream_key = key, variant["code"]
        if stream_key not in streams:
            sent, padding = encode(q["payload"], variant["code"])
            assert padding == variant["padding_bits"]
            assert len(sent) == variant["transmitted_bits"]
            streams[stream_key] = sent
        banks[variant["id"]] = {**q, "sent": streams[stream_key]}
    return banks


def receive_variant(
    received: np.ndarray, code: str, bits: int, payload_bits: int
) -> dict:
    """Recover samples using only received bits and public codebook/framing.

    Args:
        received: Received channel stream, without true error locations.
        code: Configured code name or uncoded identity.
        bits: Public quantiser label width.
        payload_bits: Real information length, excluding public ECC padding.

    Returns:
        Real decoded bits, levels and full48kHz reconstruction.
    """
    g, _ = matrices(code)
    k, n = g.shape
    expected = n * ((payload_bits + k - 1) // k)
    if len(received) != expected or payload_bits % bits or not 1 <= bits <= 16:
        raise ValueError("Received stream/public format mismatch.")
    data, _ = decode(received, code)
    data = data[:payload_bits]
    labels = bits_to_labels(data, bits)
    levels = dequantise(labels, bits, 18)
    return {
        "payload": data,
        "levels": levels,
        "high": periodic_reconstruct(levels, 96000),
    }


def measure_variant(
    system: dict,
    variant: dict,
    bank: dict,
    received: np.ndarray,
    mask: np.ndarray,
    recovered: dict,
) -> dict:
    """Measure true errors outside the receiver on the unchanged full grid.

    Args:
        system: Original high-precision reference, for measurement only.
        variant: Public code, bit-width, costs and feasibility.
        bank: True transmitter bank, never supplied to receive_variant.
        received: Received stream for a provenance hash.
        mask: Measurement-only channel flip positions.
        recovered: Receiver result produced without truth or mask.

    Returns:
        Fidelity, real-payload counts, resources, hashes and the exact
        prechannel/channel/cross-term power identity.
    """
    high = recovered["high"]
    wrong = recovered["payload"] != bank["payload"]
    pre_error = bank["before"] - system["reference"]
    channel_error = high - bank["before"]
    channel_power = float(np.mean(channel_error**2))
    cross = float(2 * np.mean(pre_error * channel_error))
    metrics = reconstruction_metrics(system["reference"], high)
    residual = abs(
        metrics["mse"] - bank["prechannel_mse"] - channel_power - cross
    )
    assert residual < 1e-9
    return {
        **metrics,
        "payload_bits": variant["payload_bits"],
        "transmitted_bits": variant["transmitted_bits"],
        "padding_bits": variant["padding_bits"],
        "logical_rate_bits_s": variant["logical_rate_bits_s"],
        "channel_flips": int(np.count_nonzero(mask)),
        "decoded_wrong_bits": int(np.count_nonzero(wrong)),
        "wrong_msb_bits": int(
            np.count_nonzero(wrong.reshape(-1, variant["bits"])[:, 0])
        ),
        "wrong_samples": int(
            np.count_nonzero(
                np.any(wrong.reshape(-1, variant["bits"]), axis=1)
            )
        ),
        "payload_perfect": int(not np.any(wrong)),
        "overload_samples": bank["overload_samples"],
        "prechannel_mse": bank["prechannel_mse"],
        "channel_error_power": channel_power,
        "channel_cross": cross,
        "power_identity_residual": residual,
        "transmitted_sha256": bits_hash(bank["sent"]),
        "received_sha256": bits_hash(received),
        "payload_sha256": bits_hash(recovered["payload"]),
        "wrong_pattern_sha256": bits_hash(wrong.astype(np.uint8)),
        "mask_sha256": bits_hash(mask.astype(np.uint8)),
    }
