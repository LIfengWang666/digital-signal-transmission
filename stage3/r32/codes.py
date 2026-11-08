"""Implement the coding example codes with GF(2) arithmetic and column matching."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Bits = NDArray[np.uint8]
CODE_NAMES = ("uncoded", "hamming74", "hamming1511", "repetition51")


def matrices(name: str) -> tuple[Bits, Bits]:
    """Construct systematic G and H using the fixed parity rows.

    Args:
        name: One of CODE_NAMES.

    Returns:
        Binary G of shape (k,n) and H of shape (n-k,n).
        The repetition H defines validity, not its majority decoder.
    """
    if name == "hamming74":
        parity = np.array(
            [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
            dtype=np.uint8,
        )
    elif name == "hamming1511":
        parity = np.array(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
    elif name == "repetition51":
        parity = np.ones((1, 4), dtype=np.uint8)
    elif name == "uncoded":
        parity = np.zeros((1, 0), dtype=np.uint8)
    else:
        raise ValueError("Unknown code name.")
    k, checks = parity.shape
    generator = np.concatenate((np.eye(k, dtype=np.uint8), parity), axis=1)
    check = np.concatenate((parity.T, np.eye(checks, dtype=np.uint8)), axis=1)
    return generator, check


def encode(payload: Bits, name: str) -> tuple[Bits, int]:
    """Pad once and encode the binary stream as in the reference.

    Args:
        payload: Flat 0/1 message bits, before ECC padding.
        name: Code name.

    Returns:
        Flat transmitted uint8 codewords and the number of appended zeros.
    """
    values = np.asarray(payload)
    if values.ndim != 1 or not np.all((values == 0) | (values == 1)):
        raise ValueError("The payload must be a flat binary array.")
    generator, _ = matrices(name)
    k = len(generator)
    padding = (-len(values)) % k
    messages = np.pad(values, (0, padding)).astype(np.uint8).reshape(-1, k)
    if name == "repetition51":
        words = np.repeat(messages, 5, axis=1)
    else:
        words = (messages @ generator) % 2
    return words.reshape(-1), padding


def decode(received: Bits, name: str) -> tuple[Bits, Bits]:
    """Decode using received bits and public code definitions only.

    Args:
        received: Flat binary words, with length divisible by n.
        name: Code name, determining the fixed matrix/majority rule.

    Returns:
        Full decoded message including padding, and corrected codewords.
        The caller removes padding using the original public payload length.
    """
    values = np.asarray(received)
    generator, check = matrices(name)
    k, n = generator.shape
    if (
        values.ndim != 1
        or len(values) % n
        or not np.all((values == 0) | (values == 1))
    ):
        raise ValueError("Use complete binary codewords.")
    words = values.astype(np.uint8).reshape(-1, n)
    if name == "repetition51":
        message = (np.sum(words, axis=1) >= 3).astype(np.uint8)
        return message, np.repeat(message[:, None], 5, axis=1).reshape(-1)
    corrected = words.copy()
    if name.startswith("hamming"):
        syndrome = (words @ check.T) % 2
        # Match actual H columns; their binary integer is not the bit index.
        for position, column in enumerate(check.T):
            matches = np.all(syndrome == column, axis=1)
            corrected[matches, position] ^= np.uint8(1)
    return corrected[:, :k].reshape(-1), corrected.reshape(-1)


def binary_words(width: int) -> Bits:
    """Enumerate all fixed-width binary row vectors, MSB first.

    Args:
        width: Positive width no greater than fifteen.

    Returns:
        Array of shape (2**width,width).
    """
    if not 1 <= width <= 15:
        raise ValueError("Enumeration is limited to widths 1 through 15.")
    values = np.arange(2**width, dtype=np.uint32)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint32)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8)


def gf2_rank(matrix: Bits) -> int:
    """Compute binary rank by XOR row elimination, not real-valued rank.

    Args:
        matrix: Two-dimensional binary matrix.

    Returns:
        Rank over GF(2).
    """
    reduced = np.asarray(matrix, dtype=np.uint8).copy()
    if reduced.ndim != 2 or not np.all((reduced == 0) | (reduced == 1)):
        raise ValueError("Use a binary matrix.")
    row = 0
    for column in range(reduced.shape[1]):
        candidates = np.flatnonzero(reduced[row:, column])
        if not len(candidates):
            continue
        pivot = row + candidates[0]
        reduced[[row, pivot]] = reduced[[pivot, row]]
        others = np.flatnonzero(reduced[:, column])
        others = others[others != row]
        reduced[others] ^= reduced[row]
        row += 1
        if row == len(reduced):
            break
    return row
