"""Implement equations (1)--(5) in the supplied degradation specification.

Callers supply the RNG explicitly. The project adopts project seed 2470964
with SeedSequence/PCG64 and purpose paths in metadata/randomness_design.json.
Severity ranges remain to be justified; no hidden severity default is used.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Bits = NDArray[np.uint8]


def add_gaussian_noise(
    samples: FloatArray, snr_db: float, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, float]:
    """Use samples/snr_db/rng to return noisy samples, noise and model
    variance.

    Noise is iid Gaussian. It is not subsequently rescaled to force the
    finite-realization SNR to equal its target exactly.

    Args:
        samples: Finite real samples in their original amplitude units.
        snr_db: Target signal-to-noise ratio in decibels.
        rng: Caller-supplied NumPy generator for independent noise
            draws.

    Returns:
        Noisy samples, the actual noise vector, and model noise
        variance.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1 or samples.size == 0 or np.iscomplexobj(samples):
        raise ValueError(
            "A nonempty, real, one-dimensional signal is required."
        )
    if not np.all(np.isfinite(samples)) or not np.isfinite(snr_db):
        raise ValueError("Samples and SNR must be finite.")
    samples = samples.astype(np.float64, copy=False)
    signal_power = float(np.mean(samples**2))
    if signal_power <= 0:
        raise ValueError(
            "A positive signal mean square is needed to define SNR."
        )
    noise_variance = signal_power / 10 ** (snr_db / 10)
    noise = rng.normal(0.0, np.sqrt(noise_variance), samples.shape)
    return samples + noise, noise, noise_variance


def transmit_bsc(
    bits: Bits, probability: float, rng: np.random.Generator
) -> tuple[Bits, NDArray[np.bool_]]:
    """Use bits/probability/rng to return received bits and a measurement-
    only mask.

    Flip positions are independent Bernoulli trials. The returned true
    mask is for evaluation and must never be supplied to a project
    decoder.

    Args:
        bits: One-dimensional array containing only binary values.
        probability: Independent bit-flip probability in the interval
            [0, 1].
        rng: Caller-supplied NumPy generator for the flip trials.

    Returns:
        Received uint8 bits and the true flip mask for measurement only.
    """
    bits = np.asarray(bits)
    if bits.ndim != 1 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Bits must be a one-dimensional binary array.")
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The crossover probability must lie in [0,1].")
    flips = rng.random(bits.shape) < probability
    return bits.astype(np.uint8, copy=False) ^ flips.astype(np.uint8), flips
