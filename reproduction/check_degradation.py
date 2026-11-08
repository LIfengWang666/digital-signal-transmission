"""Check the supplied models using separate synthetic test data."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from degradation import add_gaussian_noise, transmit_bsc

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run fixed technical checks, save CSV at ROOT/metadata, and return
    None.

    Returns:
        None. Write the described outputs or raise on invalid data.
    """
    rng = np.random.default_rng(
        0
    )  # Technical verification only; not an experimental seed.
    count = 200000
    signal = np.ones(count)
    noisy, noise, variance = add_gaussian_noise(signal, 20.0, rng)
    bits = rng.integers(0, 2, count, dtype=np.uint8)
    clean, clean_mask = transmit_bsc(bits, 0.0, rng)
    inverted, inverted_mask = transmit_bsc(bits, 1.0, rng)
    received, mask = transmit_bsc(bits, 0.05, rng)
    rows = []

    def check(
        name: str, condition: bool, measured: float, expected: float
    ) -> None:
        """Record condition and measured/expected values under name; return
        None.

        Args:
            name: Description of the mathematical property being checked.
            condition: Whether the stated check passed.
            measured: Observed numerical value recorded in CSV.
            expected: Theoretical or reference value for comparison.

        Returns:
            None. Append a result row and raise on failure.
        """
        rows.append(
            {
                "check": name,
                "passed": bool(condition),
                "measured": measured,
                "expected": expected,
            }
        )
        if not condition:
            raise ArithmeticError(name)

    check(
        "variance follows specification equation 4",
        abs(variance - 0.01) < 1e-15,
        variance,
        0.01,
    )
    check(
        "noise is added without signal normalization",
        np.array_equal(noisy, signal + noise),
        float(np.max(np.abs(noisy - signal - noise))),
        0,
    )
    check(
        "Gaussian sample mean within six model standard errors",
        abs(float(noise.mean())) <= 6 * np.sqrt(variance / count),
        float(noise.mean()),
        0,
    )
    check(
        "Gaussian sample variance within six approximate standard errors",
        abs(float(noise.var()) - variance)
        <= 6 * variance * np.sqrt(2 / (count - 1)),
        float(noise.var()),
        variance,
    )
    check(
        "BSC p0 preserves bits",
        np.array_equal(clean, bits) and not clean_mask.any(),
        float(clean_mask.mean()),
        0,
    )
    check(
        "BSC p1 flips every bit",
        np.array_equal(inverted, 1 - bits) and inverted_mask.all(),
        float(inverted_mask.mean()),
        1,
    )
    check(
        "BSC mask equals actual XOR errors",
        np.array_equal(received != bits, mask),
        float(np.mean(received != bits)),
        float(mask.mean()),
    )
    check(
        "BSC empirical rate within six standard errors",
        abs(float(mask.mean()) - 0.05) <= 6 * np.sqrt(0.05 * 0.95 / count),
        float(mask.mean()),
        0.05,
    )
    output = ROOT / "metadata"
    output.mkdir(exist_ok=True)
    with (output / "degradation_model_checks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "Eight degradation-model technical checks passed; "
        "R1.1 remains noise-free."
    )


if __name__ == "__main__":
    main()
