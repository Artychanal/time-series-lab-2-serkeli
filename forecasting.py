from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nonlinear_model import event_model
from polynomial_lsm import predict_log


@dataclass(frozen=True)
class ForecastBundle:
    frame: pd.DataFrame
    horizon: int


def _bootstrap_intervals(
    point_log: np.ndarray,
    residuals: np.ndarray,
    simulations: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(np.asarray(residuals, dtype=float), size=simulations, replace=True)[:, None]
    simulated = np.maximum(0.0, np.expm1(point_log[None, :] + sampled))
    return np.quantile(simulated, 0.025, axis=0), np.quantile(simulated, 0.975, axis=0)


def build_forecasts(
    dates: pd.Series,
    observed_log: np.ndarray,
    polynomial_degree: int,
    polynomial_coefficients: np.ndarray,
    polynomial_scale: float,
    polynomial_fitted: np.ndarray,
    nonlinear_parameters: np.ndarray,
    nonlinear_scale: float,
    nonlinear_fitted: np.ndarray,
    phantom_index: int,
    seed: int = 42,
) -> ForecastBundle:
    n = len(observed_log)
    horizon = int(np.ceil(0.5 * n))
    future_t = np.arange(n, n + horizon, dtype=float)
    future_dates = pd.date_range(pd.Timestamp(dates.iloc[-1]) + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    polynomial_log = predict_log(
        polynomial_coefficients, future_t, polynomial_degree, polynomial_scale
    )
    nonlinear_log = event_model(future_t, nonlinear_parameters, nonlinear_scale, phantom_index)
    polynomial_point = np.maximum(0.0, np.expm1(polynomial_log))
    nonlinear_point = np.maximum(0.0, np.expm1(nonlinear_log))
    polynomial_lower, polynomial_upper = _bootstrap_intervals(
        polynomial_log, np.asarray(observed_log) - np.asarray(polynomial_fitted), seed=seed
    )
    nonlinear_lower, nonlinear_upper = _bootstrap_intervals(
        nonlinear_log, np.asarray(observed_log) - np.asarray(nonlinear_fitted), seed=seed + 1
    )
    frame = pd.DataFrame({
        "date": future_dates,
        "polynomial_forecast": polynomial_point,
        "polynomial_lower_95": polynomial_lower,
        "polynomial_upper_95": polynomial_upper,
        "nonlinear_forecast": nonlinear_point,
        "nonlinear_lower_95": nonlinear_lower,
        "nonlinear_upper_95": nonlinear_upper,
    })
    return ForecastBundle(frame=frame, horizon=horizon)
