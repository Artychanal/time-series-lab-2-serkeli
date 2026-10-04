from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from polynomial_lsm import smape


@dataclass(frozen=True)
class NonlinearResult:
    parameters: tuple[float, ...]
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
    phantom_index: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["parameters"] = list(self.parameters)
        return data


def event_model(
    t: np.ndarray,
    parameters: np.ndarray,
    scale: float,
    phantom_index: int,
) -> np.ndarray:
    beta0, beta1, amplitude_release, decay_release, amplitude_phantom, decay_phantom = parameters
    tau = np.asarray(t, dtype=float) / max(scale, 1.0)
    phantom_tau = phantom_index / max(scale, 1.0)
    after_phantom = tau >= phantom_tau
    release_component = amplitude_release * np.exp(-decay_release * tau)
    phantom_component = np.where(
        after_phantom,
        amplitude_phantom * np.exp(-decay_phantom * (tau - phantom_tau)),
        0.0,
    )
    return beta0 + beta1 * tau + release_component + phantom_component


def _initial_parameters(values: np.ndarray, phantom_index: int) -> np.ndarray:
    tail = values[max(1, len(values) // 2):]
    beta0 = float(np.median(tail))
    release_amplitude = max(float(values[0] - beta0), 0.1)
    if phantom_index < len(values):
        local_start = max(0, phantom_index - 3)
        local_level = float(np.median(values[local_start:phantom_index]))
        phantom_amplitude = max(float(values[phantom_index] - local_level), 0.1)
    else:
        phantom_amplitude = 0.1
    return np.array([beta0, 0.0, release_amplitude, 8.0, phantom_amplitude, 12.0])


def fit_event_model(
    log_values: np.ndarray,
    phantom_index: int,
    scale: float | None = None,
    n_starts: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, float, np.ndarray]:
    values = np.asarray(log_values, dtype=float)
    used_scale = max(float(len(values) - 1) if scale is None else float(scale), 1.0)
    t = np.arange(len(values), dtype=float)
    base = _initial_parameters(values, phantom_index)
    lower = np.array([-20.0, -20.0, 0.0, 0.0, 0.0, 0.0])
    upper = np.array([30.0, 20.0, 20.0, 100.0, 20.0, 100.0])
    rng = np.random.default_rng(seed)
    starts = [base]
    for _ in range(max(1, n_starts) - 1):
        candidate = base.copy()
        candidate[:2] += rng.normal(0.0, [0.5, 0.5])
        candidate[2] *= rng.uniform(0.5, 1.8)
        candidate[3] *= rng.uniform(0.35, 2.5)
        candidate[4] *= rng.uniform(0.5, 1.8)
        candidate[5] *= rng.uniform(0.35, 2.5)
        starts.append(np.clip(candidate, lower + 1e-8, upper - 1e-8))

    best = None
    for start in starts:
        fitted = least_squares(
            lambda params: event_model(t, params, used_scale, phantom_index) - values,
            x0=np.clip(start, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.25,
            max_nfev=20_000,
        )
        rss = float(np.sum((event_model(t, fitted.x, used_scale, phantom_index) - values) ** 2))
        if best is None or rss < best[0]:
            best = (rss, fitted.x)
    assert best is not None
    parameters = np.asarray(best[1], dtype=float)
    return parameters, used_scale, event_model(t, parameters, used_scale, phantom_index)


def expanding_window_event_predictions(
    log_values: np.ndarray,
    phantom_index: int,
    min_train: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(log_values, dtype=float)
    first_train = max(24, phantom_index + 6) if min_train is None else min_train
    actual: list[float] = []
    predicted: list[float] = []
    for end in range(first_train, len(values)):
        params, scale, _ = fit_event_model(
            values[:end], phantom_index=phantom_index, n_starts=2, seed=42 + end
        )
        estimate = event_model(np.array([end]), params, scale, phantom_index)[0]
        actual.append(float(values[end]))
        predicted.append(float(estimate))
    return np.asarray(actual), np.asarray(predicted)


def evaluate_event_model(log_values: np.ndarray, phantom_index: int) -> tuple[NonlinearResult, np.ndarray]:
    values = np.asarray(log_values, dtype=float)
    parameters, scale, fitted = fit_event_model(values, phantom_index=phantom_index)
    actual_original = np.expm1(values)
    fitted_original = np.maximum(0.0, np.expm1(fitted))
    cv_actual, cv_predicted = expanding_window_event_predictions(values, phantom_index)
    residuals = values - fitted
    rss = max(float(np.sum(residuals**2)), 1e-12)
    n = len(values)
    k = len(parameters)
    sigma2 = rss / n
    result = NonlinearResult(
        parameters=tuple(float(value) for value in parameters),
        r2_log=float(r2_score(values, fitted)),
        rmse_log=float(np.sqrt(mean_squared_error(values, fitted))),
        mae_log=float(mean_absolute_error(values, fitted)),
        rmse_original=float(np.sqrt(mean_squared_error(actual_original, fitted_original))),
        mae_original=float(mean_absolute_error(actual_original, fitted_original)),
        smape_original=smape(actual_original, fitted_original),
        cv_rmse_log=float(np.sqrt(mean_squared_error(cv_actual, cv_predicted))),
        cv_mae_log=float(mean_absolute_error(cv_actual, cv_predicted)),
        cv_smape_original=smape(np.expm1(cv_actual), np.maximum(0.0, np.expm1(cv_predicted))),
        aic=float(n * np.log(sigma2) + 2 * k),
        bic=float(n * np.log(sigma2) + k * np.log(n)),
        phantom_index=phantom_index,
    )
    return result, fitted
