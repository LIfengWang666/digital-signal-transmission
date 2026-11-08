"""Independently check the DFT folding identity for every measured rate."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Compare transformed coarse samples with folded original DFT bins.

    Returns:
        None. Save five independent numerical checks to metadata;
        raise if normalised complex coefficients disagree.
    """
    reference = np.load(ROOT / "reference_signal.npy", allow_pickle=False)
    count = reference.size
    transformed = np.fft.fft(reference, norm="backward")
    with (ROOT / "stage1/r14/results/decision_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        candidates = list(csv.DictReader(stream))
    rows = []
    for candidate in candidates:
        sample_count = int(candidate["sample_count"])
        if count % sample_count:
            raise ValueError("The identity requires L=M*N with integer M.")
        factor = count // sample_count
        samples = reference[::factor]
        # These two routes use the complete complex spectra, including phase.
        measured = np.fft.fft(samples, norm="backward") / sample_count
        predicted = transformed.reshape(factor, sample_count).sum(axis=0)
        predicted /= count
        error = float(np.max(np.abs(measured - predicted)))
        tolerance = 1e-12
        passed = bool(np.isfinite(error) and error <= tolerance)
        rows.append(
            {
                "sample_rate_hz": float(candidate["sample_rate_hz"]),
                "factor": factor,
                "compared_complex_bins": sample_count,
                "maximum_coefficient_error": error,
                "tolerance": tolerance,
                "unit": "NPY units",
                "passed": passed,
            }
        )
    destination = ROOT / "metadata/sampling_identity_checks.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not all(row["passed"] for row in rows):
        raise ArithmeticError(
            "The measured spectra fail the folding identity."
        )
    print("Five full-complex-spectrum folding identity checks passed.")


if __name__ == "__main__":
    main()
