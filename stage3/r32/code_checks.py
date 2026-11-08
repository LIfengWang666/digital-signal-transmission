"""Verify coding example examples, exact code capability and all error patterns."""

from __future__ import annotations

import json
from itertools import combinations
from math import comb, log2
from pathlib import Path

import numpy as np

from stage2.r21.filtering_approach import write_csv
from stage3.r32.codes import (
    CODE_NAMES,
    binary_words,
    decode,
    encode,
    gf2_rank,
    matrices,
)

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Enumerate codewords and decoder residuals before any BSC experiment.

    Returns:
        None. Save code definitions, exact counts, reference examples and checks.
    """
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    specifications, coefficients, checks, examples = [], [], [], []
    for name in CODE_NAMES:
        generator, check = matrices(name)
        k, n = generator.shape
        messages = binary_words(k)
        encoded, _ = encode(messages.reshape(-1), name)
        words = encoded.reshape(-1, n)
        d_min = int(np.min(np.sum(words[1:], axis=1)))
        t = (d_min - 1) // 2
        assert gf2_rank(generator) == k
        assert gf2_rank(check) == n - k
        assert not np.any((generator @ check.T) % 2)
        assert not np.any((words @ check.T) % 2)
        assert d_min == (
            5 if name == "repetition51" else 1 if name == "uncoded" else 3
        )
        if name.startswith("hamming"):
            assert np.all(np.any(check, axis=0))
            assert len(np.unique(check.T, axis=0)) == n
        no_error, _ = decode(encoded, name)
        assert np.array_equal(no_error.reshape(-1, k), messages)
        single_cases = 0
        if t >= 1:
            for position in range(n):
                corrupted = words.copy()
                corrupted[:, position] ^= 1
                recovered, corrected = decode(corrupted.reshape(-1), name)
                assert np.array_equal(recovered.reshape(-1, k), messages)
                assert np.array_equal(corrected.reshape(-1, n), words)
                single_cases += len(messages)
        errors = binary_words(n)
        decoded, corrected = decode(errors.reshape(-1), name)
        recovered = decoded.reshape(-1, k)
        residual = corrected.reshape(-1, n)
        assert not np.any((residual @ check.T) % 2)
        weights = np.sum(errors, axis=1)
        wrong = np.sum(recovered, axis=1)
        assert not np.any(wrong[weights <= t])
        assert np.all(wrong[weights > t] > 0)
        # Translation invariance follows from s=(c+e)H^T=eH^T.
        # The small code additionally checks every message/error combination.
        translation_cases = 0
        if n <= 7:
            for word, message in zip(words, messages):
                translated, _ = decode((errors ^ word).reshape(-1), name)
                assert np.array_equal(
                    translated.reshape(-1, k) ^ message, recovered
                )
                translation_cases += len(errors)
        for weight in range(n + 1):
            subset = weights == weight
            coefficients.append(
                {
                    "code": name,
                    "weight": weight,
                    "patterns": int(np.count_nonzero(subset)),
                    "wrong_message_bits_sum": int(np.sum(wrong[subset])),
                    "failed_message_blocks": int(
                        np.count_nonzero(wrong[subset])
                    ),
                }
            )
            assert np.count_nonzero(subset) == comb(n, weight)
        for length in [1, 3, 14, 31, 448000]:
            payload = (np.arange(length) % 3 == 0).astype(np.uint8)
            transmitted, padding = encode(payload, name)
            recovered_bits, _ = decode(transmitted, name)
            assert np.array_equal(recovered_bits[:length], payload)
            assert padding == (-length) % k
            assert len(transmitted) == n * ((length + k - 1) // k)
        specifications.append(
            {
                "code": name,
                "n": n,
                "k": k,
                "rate": k / n,
                "d_min": d_min,
                "t": t,
                "G": generator.tolist(),
                "H": check.tolist(),
                "P": generator[:, k:].tolist(),
            }
        )
        checks.append(
            {
                "code": name,
                "all_message_count": len(messages),
                "single_error_cases": single_cases,
                "all_error_patterns": len(errors),
                "translation_cases": translation_cases,
                "gf2_rank_G": k,
                "gf2_rank_H": n - k,
                "d_min": d_min,
                "t": t,
                "all_checks_passed": True,
            }
        )
    for name, message, bit in [
        ("hamming74", [1, 1, 0, 0], 2),
        ("hamming1511", [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0], 7),
        ("hamming74", [1, 0, 1, 1], 6),
    ]:
        sent, _ = encode(np.array(message, dtype=np.uint8), name)
        received = sent.copy()
        received[bit] ^= 1
        _, check = matrices(name)
        syndrome = (received @ check.T) % 2
        recovered, corrected = decode(received, name)
        assert np.array_equal(recovered, message)
        examples.append(
            {
                "code": name,
                "message": "".join(map(str, message)),
                "encoded": "".join(map(str, sent)),
                "flipped_zero_based_position": bit,
                "received": "".join(map(str, received)),
                "syndrome": "".join(map(str, syndrome)),
                "corrected": "".join(map(str, corrected)),
                "decoded": "".join(map(str, recovered)),
            }
        )
    given = [
        np.array(v)
        for v in [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
        ]
    ]
    distances = [
        {
            "first": i + 1,
            "second": j + 1,
            "distance": int(np.count_nonzero(given[i] != given[j])),
        }
        for i, j in combinations(range(4), 2)
    ]
    assert [r["distance"] for r in distances] == [3, 3, 4, 4, 3, 3]
    error = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
    recovered, corrected = decode(error, "hamming74")
    _, check = matrices("hamming74")
    assert np.count_nonzero(recovered) == 3
    assert not np.any(corrected @ check.T % 2)
    probability = 0.001
    rep_ber = sum(
        comb(5, j) * probability**j * (1 - probability) ** (5 - j)
        for j in range(3, 6)
    )
    entropy = -probability * log2(probability) - (1 - probability) * log2(
        1 - probability
    )
    write_csv(results / "code_checks.csv", checks)
    write_csv(results / "error_pattern_coefficients.csv", coefficients)
    write_csv(results / "lab_examples.csv", examples)
    write_csv(results / "lab_pairwise_distances.csv", distances)
    (results / "code_definitions.json").write_text(
        json.dumps(specifications, indent=2) + "\n"
    )
    (results / "lab_and_multiflip_checks.json").write_text(
        json.dumps(
            {
                "all_checks_passed": True,
                "lab_repetition_probability": probability,
                "lab_repetition_decoded_ber": rep_ber,
                "lab_bsc_capacity": 1 - entropy,
                "double_error_received": error.tolist(),
                "double_error_decoded": recovered.tolist(),
                "double_error_corrected": corrected.tolist(),
                "zero_post_syndrome_does_not_imply_recovery": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("R3.2: reference matrices, code capability and all patterns verified.")


if __name__ == "__main__":
    main()
