from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class PolynomialResult:
    degree: int
    coefficients: tuple[float, ...]
    r2_log: float
    rmse_log: float
    mae_log: float
    rmse_original: float
    mae_original: float
    smape_original: float
    cv_rmse_log: float
    cv_mae_log: float
    cv_smape_original: float
    aic: float
    bic: float

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["coefficients"] = list(self.coefficients)
        return data


def design_matrix(t: np.ndarray, degree: int, scale: float) -> np.ndarray:
    tau = np.asarray(t, dtype=float) / max(float(scale), 1.0)
    return np.column_stack([tau**power for power in range(degree + 1)])


def fit_coefficients(log_values: np.ndarray, degree: int, scale: float | None = None) -> tuple[np.ndarray, float]:
    values = np.asarray(log_values, dtype=float)
    t = np.arange(len(values), dtype=float)
    used_scale = max(float(len(values) - 1) if scale is None else float(scale), 1.0)
    coefficients, *_ = np.linalg.lstsq(design_matrix(t, degree, used_scale), values, rcond=None)
    return coefficients, used_scale


def predict_log(coefficients: np.ndarray, t: np.ndarray, degree: int, scale: float) -> np.ndarray:
    return design_matrix(np.asarray(t, dtype=float), degree, scale) @ np.asarray(coefficients)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual) + np.abs(predicted)
    terms = np.divide(2.0 * np.abs(predicted - actual), denominator, out=np.zeros_like(actual), where=denominator > 0)
    return float(np.mean(terms) * 100.0)


def expanding_window_predictions(
    log_values: np.ndarray, degree: int, min_train: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    actual: list[float] = []
    predicted: list[float] = []
    for end in range(min_train, len(log_values)):
        coefficients, scale = fit_coefficients(log_values[:end], degree)
        estimate = predict_log(coefficients, np.array([end]), degree, scale)[0]
        actual.append(float(log_values[end]))
        predicted.append(float(estimate))
    return np.asarray(actual), np.asarray(predicted)


def evaluate_polynomial(log_values: np.ndarray, degree: int, min_train: int = 24) -> tuple[PolynomialResult, np.ndarray]:
    values = np.asarray(log_values, dtype=float)
    coefficients, scale = fit_coefficients(values, degree)
    fitted = predict_log(coefficients, np.arange(len(values)), degree, scale)
    actual_original = np.expm1(values)
    fitted_original = np.maximum(0.0, np.expm1(fitted))
    residuals = values - fitted
    rss = max(float(np.sum(residuals**2)), 1e-12)
    n = len(values)
    k = len(coefficients)
    sigma2 = rss / n

    cv_actual, cv_predicted = expanding_window_predictions(values, degree, min_train=min_train)
    cv_actual_original = np.expm1(cv_actual)
    cv_predicted_original = np.maximum(0.0, np.expm1(cv_predicted))
    result = PolynomialResult(
        degree=degree,
        coefficients=tuple(float(value) for value in coefficients),
        r2_log=float(r2_score(values, fitted)),
        rmse_log=float(np.sqrt(mean_squared_error(values, fitted))),
        mae_log=float(mean_absolute_error(values, fitted)),
        rmse_original=float(np.sqrt(mean_squared_error(actual_original, fitted_original))),
        mae_original=float(mean_absolute_error(actual_original, fitted_original)),
        smape_original=smape(actual_original, fitted_original),
        cv_rmse_log=float(np.sqrt(mean_squared_error(cv_actual, cv_predicted))),
        cv_mae_log=float(mean_absolute_error(cv_actual, cv_predicted)),
        cv_smape_original=smape(cv_actual_original, cv_predicted_original),
        aic=float(n * np.log(sigma2) + 2 * k),
        bic=float(n * np.log(sigma2) + k * np.log(n)),
    )
    return result, fitted
