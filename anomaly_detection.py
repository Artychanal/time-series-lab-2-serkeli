from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


MAD_SCALE = 1.4826


@dataclass(frozen=True)
class AnomalyValidation:
    injected_indices: tuple[int, ...]
    detected_indices: tuple[int, ...]
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def polynomial_baseline(values: np.ndarray, degree: int = 2) -> np.ndarray:
    t = np.linspace(0.0, 1.0, len(values))
    matrix = np.column_stack([t**power for power in range(degree + 1)])
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    return matrix @ coefficients


def hampel_on_residuals(
    values: np.ndarray,
    window: int = 5,
    threshold: float = 3.5,
    protected_indices: set[int] | None = None,
) -> pd.DataFrame:
    if window < 1:
        raise ValueError("window must be positive")
    protected = protected_indices or set()
    residuals = np.asarray(values, dtype=float) - polynomial_baseline(np.asarray(values, dtype=float))
    n = len(residuals)
    rows: list[dict[str, float | int | bool]] = []
    for index, value in enumerate(residuals):
        left = max(0, index - window)
        right = min(n, index + window + 1)
        neighborhood = np.concatenate([residuals[left:index], residuals[index + 1:right]])
        if neighborhood.size < 3:
            neighborhood = residuals[left:right]
        local_median = float(np.median(neighborhood))
        mad = float(np.median(np.abs(neighborhood - local_median)))
        robust_sigma = max(MAD_SCALE * mad, 1e-9)
        score = abs(float(value) - local_median) / robust_sigma
        flagged = bool(score > threshold)
        is_protected = index in protected
        rows.append({
            "index": index,
            "residual": float(value),
            "local_median": local_median,
            "mad": mad,
            "robust_z_score": score,
            "statistical_anomaly": flagged,
            "protected_event": is_protected,
            "correction_candidate": flagged and not is_protected,
        })
    return pd.DataFrame(rows)


def clean_with_pchip(values: np.ndarray, correction_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mask = np.asarray(correction_mask, dtype=bool)
    if mask.size != values.size:
        raise ValueError("correction mask and values must have the same length")
    if not mask.any():
        return values.copy()
    valid = ~mask
    if valid.sum() < 2:
        raise ValueError("At least two valid points are required for interpolation")
    indices = np.arange(len(values), dtype=float)
    interpolator = PchipInterpolator(indices[valid], values[valid], extrapolate=True)
    cleaned = values.copy()
    cleaned[mask] = interpolator(indices[mask])
    return cleaned


def validate_detector_with_injected_anomalies(
    clean_values: np.ndarray,
    injected_indices: tuple[int, ...] = (14, 24, 31, 52, 60),
    shifts: tuple[float, ...] = (1.4, -2.0, -2.2, 1.3, 1.8),
    window: int = 5,
    threshold: float = 3.5,
) -> tuple[np.ndarray, pd.DataFrame, AnomalyValidation]:
    contaminated = np.asarray(clean_values, dtype=float).copy()
    if len(injected_indices) != len(shifts):
        raise ValueError("Every injected index must have a corresponding shift")
    for index, shift in zip(injected_indices, shifts, strict=True):
        if not 0 <= index < len(contaminated):
            raise IndexError(f"Injected anomaly index {index} is outside the series")
        contaminated[index] += shift

    baseline_detection = hampel_on_residuals(
        np.asarray(clean_values, dtype=float), window=window, threshold=threshold
    )
    baseline_flags = set(
        baseline_detection.loc[baseline_detection["statistical_anomaly"], "index"].astype(int)
    )
    detection = hampel_on_residuals(contaminated, window=window, threshold=threshold)
    all_contaminated_flags = set(
        detection.loc[detection["statistical_anomaly"], "index"].astype(int)
    )
    detected = all_contaminated_flags - baseline_flags
    detection["baseline_statistical_anomaly"] = detection["index"].isin(baseline_flags)
    detection["new_after_injection"] = detection["index"].isin(detected)
    expected = set(injected_indices)
    tp = len(detected & expected)
    fp = len(detected - expected)
    fn = len(expected - detected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = AnomalyValidation(
        injected_indices=injected_indices,
        detected_indices=tuple(sorted(detected)),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )
    return contaminated, detection, result
