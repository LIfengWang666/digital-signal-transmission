"""Map fixed-width unsigned labels to bits using explicit NumPy operations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Indices = NDArray[np.uint32]
Bits = NDArray[np.uint8]


def labels_to_bits(labels: Indices, bits: int) -> Bits:
    """Serialise labels sample by sample, most significant bit first.

    Args:
        labels: One-dimensional integer labels in [0, 2**bits - 1].
        bits: Fixed label width, between one and sixteen.

    Returns:
        A flat uint8 array with exactly len(labels) * bits binary values.
    """
    values = np.asarray(labels)
    if not isinstance(bits, (int, np.integer)) or not 1 <= bits <= 16:
        raise ValueError("The supported label width is 1 through 16.")
    if (
        values.ndim != 1
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
        or np.any(values >= 2**bits)
    ):
        raise ValueError("Labels must be one-dimensional unsigned values.")
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    fields = (values.astype(np.uint32)[:, None] >> shifts) & 1
    return fields.astype(np.uint8).reshape(-1)


def bits_to_labels(stream: Bits, bits: int) -> Indices:
    """Recover labels from complete MSB-first fields without padding.

    Args:
        stream: One-dimensional binary values, length divisible by bits.
        bits: The same fixed width used by labels_to_bits.

    Returns:
        A uint32 label per complete field. ECC padding is not handled here.
    """
    values = np.asarray(stream)
    if not isinstance(bits, (int, np.integer)) or not 1 <= bits <= 16:
        raise ValueError("The supported label width is 1 through 16.")
    if (
        values.ndim != 1
        or len(values) % bits
        or not np.all((values == 0) | (values == 1))
    ):
        raise ValueError("Binary fields must have a complete fixed width.")
    weights = 1 << np.arange(bits - 1, -1, -1, dtype=np.uint32)
    return values.astype(np.uint32).reshape(-1, bits) @ weights
