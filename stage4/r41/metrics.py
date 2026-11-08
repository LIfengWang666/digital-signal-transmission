"""Measure reconstructed records using the reference mean-square model."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def db_from_nmse(nmse: float) -> float:
    """Express a nonnegative relative error power as a reference/error ratio.

    Args:
        nmse: MSE divided by the fixed, positive reference mean square.

    Returns:
        Minus ten log10 of NMSE, or positive infinity for exact zero error.
    """
    if not math.isfinite(nmse) or nmse < 0:
        raise ValueError("NMSE must be finite and nonnegative.")
    return math.inf if nmse == 0 else -10.0 * math.log10(nmse)


def reconstruction_metrics(
    reference: FloatArray, reconstructed: FloatArray
) -> dict[str, float]:
    """Compare two real records on their existing common time/amplitude grid.

    Args:
        reference: Nonempty finite one-dimensional reference with power > 0.
        reconstructed: Finite real samples with exactly the reference shape.

    Returns:
        MSE, NMSE, MAE, RMSE, error bias/variance, peak absolute error,
        reference/error ratio in dB and diagnostic Pearson correlation.
        Correlation is NaN when either centred record has zero power.
        No fitting, alignment, endpoint removal or epsilon floor is used.
    """
    if np.iscomplexobj(reference) or np.iscomplexobj(reconstructed):
        raise ValueError("The project metric expects real records.")
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(reconstructed, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size == 0:
        raise ValueError(
            "Use nonempty one-dimensional records of equal shape."
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Both records must contain only finite samples.")
    error = y - x
    power = float(np.mean(x**2))
    if not math.isfinite(power) or power <= 0:
        raise ValueError("NMSE requires finite positive reference power.")
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    if not math.isfinite(mse) or not math.isfinite(mae):
        raise ValueError("Error statistics exceed the finite numeric range.")
    nmse = mse / power
    bias = float(np.mean(error))
    centred_error = error - bias
    xc, yc = x - np.mean(x), y - np.mean(y)
    px, py = float(np.mean(xc**2)), float(np.mean(yc**2))
    correlation = math.nan
    if px > 0 and py > 0:
        correlation = float(np.mean(xc * yc) / (math.sqrt(px) * math.sqrt(py)))
        correlation = float(np.clip(correlation, -1.0, 1.0))
    return {
        "samples": int(x.size),
        "reference_power": power,
        "mse": mse,
        "nmse": nmse,
        "mae": mae,
        "rmse": math.sqrt(mse),
        "mean_error": bias,
        "error_variance_ddof0": float(np.mean(centred_error**2)),
        "peak_absolute_error": float(np.max(np.abs(error))),
        "reconstruction_error_ratio_db": db_from_nmse(nmse),
        "correlation": correlation,
    }
