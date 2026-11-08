"""Verify both supplied references without changing either file."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "reference_signal.npy": (
        "2f62c84fb0e11586373b2dc0d92f180206" "d15d4a77098046c9c43de290657095"
    ),
    "reference_signal.wav": (
        "0ffec41d7773f8a5fda064aecfec3affe9" "56b6955227ab0c5774e4a4349c9454"
    ),
}


def main() -> None:
    """Check files in ROOT, save their metadata as CSV, and return None.

    Returns:
        None. Write the described outputs or raise on invalid data.
    """
    output = ROOT / "metadata"
    output.mkdir(exist_ok=True)
    rows = []
    for name, expected in EXPECTED_HASHES.items():
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"The supplied input has changed: {name}")
        rows.append(
            {"file": name, "sha256": actual, "bytes": path.stat().st_size}
        )
    with (output / "provided_files.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    samples = np.load(ROOT / "reference_signal.npy", allow_pickle=False)
    rate, pcm = wavfile.read(ROOT / "reference_signal.wav")
    if (
        samples.shape != (96000,)
        or pcm.shape != samples.shape
        or rate != 48000
    ):
        raise ValueError(
            "Reference dimensions or WAV rate do not match the supplied data."
        )
    if not np.all(np.isfinite(samples)) or pcm.dtype != np.int16:
        raise ValueError("Unexpected reference values or WAV storage type.")
    # This measured file relationship is used only for the input cross-check.
    # It does not define later project quantization or modify NPY data.
    if not np.array_equal(np.rint(20000 * samples).astype(np.int16), pcm):
        raise ValueError("The two reference representations no longer agree.")
    residual = pcm.astype(float) - 20000 * samples
    facts = {
        "npy_samples": samples.size,
        "npy_storage": str(samples.dtype),
        "wav_sample_rate_hz": rate,
        "wav_storage": str(pcm.dtype),
        "wav_channels": 1,
        "pcm_counts_per_npy_unit": 20000,
        "nearest_integer_mapping_exact": True,
        "pcm_mapping_maximum_residual_counts": float(np.max(np.abs(residual))),
        "pcm_mapping_rms_residual_counts": float(
            np.sqrt(np.mean(residual**2))
        ),
    }
    with (output / "provided_signal_properties.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["property", "value"])
        writer.writerows(facts.items())
    print("Original NPY/WAV hashes and PCM mapping verified.")


if __name__ == "__main__":
    main()
