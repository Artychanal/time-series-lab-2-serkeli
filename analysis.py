from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from anomaly_detection import (
    clean_with_pchip,
    hampel_on_residuals,
    validate_detector_with_injected_anomalies,
)
from data_loader import load_and_validate_monthly_reviews
from forecasting import build_forecasts
from nonlinear_model import evaluate_event_model, event_model, fit_event_model
from polynomial_lsm import evaluate_polynomial, fit_coefficients, predict_log, smape
from visualization import create_figures


RANDOM_SEED = 42
PHANTOM_DATE = pd.Timestamp("2023-09-01")
PROTECTED_EVENT_MONTHS = {
    "2020-12",  # release
    "2023-09", "2023-10", "2023-11", "2023-12",  # Phantom Liberty period
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _backtest(
    log_values: np.ndarray,
    polynomial_degrees: tuple[int, ...],
    phantom_index: int,
    test_size: int = 12,
) -> pd.DataFrame:
    train = log_values[:-test_size]
    actual_log = log_values[-test_size:]
    future_t = np.arange(len(train), len(log_values), dtype=float)

    nonlinear_parameters, nonlinear_scale, _ = fit_event_model(
        train, phantom_index=phantom_index, n_starts=6, seed=RANDOM_SEED
    )
    nonlinear_predicted = event_model(
        future_t, nonlinear_parameters, nonlinear_scale, phantom_index
    )

    actual = np.expm1(actual_log)
    predictions: list[tuple[str, np.ndarray]] = []
    for degree in polynomial_degrees:
        coefficients, scale = fit_coefficients(train, degree)
        predictions.append(
            (f"Поліном {degree}", predict_log(coefficients, future_t, degree, scale))
        )
    predictions.append(("Подієва R&D", nonlinear_predicted))

    rows = []
    for name, predicted_log in predictions:
        predicted = np.maximum(0.0, np.expm1(predicted_log))
        rows.append({
            "model": name,
            "test_months": test_size,
            "backtest_rmse_log": float(np.sqrt(mean_squared_error(actual_log, predicted_log))),
            "backtest_mae_log": float(mean_absolute_error(actual_log, predicted_log)),
            "backtest_rmse_original": float(np.sqrt(mean_squared_error(actual, predicted))),
            "backtest_mae_original": float(mean_absolute_error(actual, predicted)),
            "backtest_smape_original": smape(actual, predicted),
        })
    return pd.DataFrame(rows)


def _print_summary(summary: dict[str, Any]) -> None:
    quality = summary["data_quality"]
    anomalies = summary["anomalies"]
    validation = summary["detector_validation"]
    cleaning = summary["controlled_cleaning"]

    print("\nВХІДНІ ДАНІ")
    print(f"Спостережень: {quality['observations']} із {quality['raw_observations']}")
    print(f"Період: {quality['period_start']} — {quality['period_end']}")
    print(
        "Пропуски / дублікати / порушення сум: "
        f"{quality['missing_values']} / {quality['duplicate_months']} / "
        f"{0 if quality['totals_consistent'] else 1}"
    )

    print("\nАНОМАЛІЇ ТА ОЧИЩЕННЯ")
    print(f"Оптимальний поріг Hampel: {anomalies['threshold']:.1f}")
    print(
        f"Precision: {validation['precision']:.3f}; "
        f"Recall: {validation['recall']:.3f}; F1: {validation['f1']:.3f}"
    )
    print(f"Статистичних аномалій у реальному ряді: {anomalies['statistical_count']}")
    print(
        f"PCHIP: скориговано {cleaning['corrected_count']} точки; "
        f"RMSE(log) {cleaning['rmse_before_log']:.4f} -> {cleaning['rmse_after_log']:.4f}"
    )

    print("\nПОРІВНЯННЯ МОДЕЛЕЙ")
    models = pd.DataFrame(summary["models"])[[
        "model",
        "r2_log",
        "rmse_log",
        "cv_rmse_log",
        "cv_smape_original",
        "backtest_rmse_log",
        "backtest_smape_original",
    ]].rename(columns={
        "model": "Модель",
        "r2_log": "R2 log",
        "rmse_log": "RMSE log",
        "cv_rmse_log": "CV RMSE",
        "cv_smape_original": "CV sMAPE, %",
        "backtest_rmse_log": "BT RMSE",
        "backtest_smape_original": "BT sMAPE, %",
    })
    print(models.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Найкращий поліном за CV RMSE: степінь {summary['selected_polynomial_degree']}")

    print("\nПРОГНОЗ")
    print(
        f"Горизонт: {summary['forecast_horizon']} місяців; "
        f"період: {summary['forecast_start']} — {summary['forecast_end']}"
    )
    print("95% bootstrap-інтервали сформовано для обох моделей.")


def run_analysis(data_dir: Path, results_dir: Path) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data, quality = load_and_validate_monthly_reviews(
        data_dir / "cyberpunk_2077_monthly_reviews.csv"
    )
    log_values = data["log_total_reviews"].to_numpy(float)
    protected_indices = {
        index for index, date in enumerate(data["date"])
        if date.strftime("%Y-%m") in PROTECTED_EVENT_MONTHS
    }

    threshold_candidates = (2.5, 3.0, 3.5, 4.0)
    sensitivity_rows = []
    validation_runs = []
    for threshold in threshold_candidates:
        candidate = hampel_on_residuals(
            log_values, window=5, threshold=threshold, protected_indices=protected_indices
        )
        contaminated_candidate, injected_candidate, validation = (
            validate_detector_with_injected_anomalies(log_values, threshold=threshold)
        )
        sensitivity_rows.append({
            "threshold": threshold,
            "statistical_anomalies": int(candidate["statistical_anomaly"].sum()),
            "protected_events": int((candidate["statistical_anomaly"] & candidate["protected_event"]).sum()),
            "correction_candidates": int(candidate["correction_candidate"].sum()),
            "precision": validation.precision,
            "recall": validation.recall,
            "f1": validation.f1,
        })
        validation_runs.append(
            (threshold, contaminated_candidate, injected_candidate, validation)
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    selected_threshold, contaminated, injected_detection, detector_validation = max(
        validation_runs,
        key=lambda item: (item[3].f1, item[3].recall, item[3].precision, -item[0]),
    )

    initial_anomalies = hampel_on_residuals(
        log_values, window=5, threshold=selected_threshold, protected_indices=protected_indices
    )
    verified_real_indices = set(
        initial_anomalies.loc[initial_anomalies["statistical_anomaly"], "index"].astype(int)
    )
    anomalies = hampel_on_residuals(
        log_values,
        window=5,
        threshold=selected_threshold,
        protected_indices=protected_indices | verified_real_indices,
    )
    anomalies.insert(1, "date", data["date"].dt.strftime("%Y-%m-%d"))
    correction_mask = anomalies["correction_candidate"].to_numpy(bool)
    cleaned_log = clean_with_pchip(log_values, correction_mask)
    data["cleaned_log_total_reviews"] = cleaned_log
    data["cleaned_total_reviews"] = np.maximum(0.0, np.expm1(cleaned_log))

    controlled_correction_mask = injected_detection["new_after_injection"].to_numpy(bool)
    controlled_cleaned = clean_with_pchip(contaminated, controlled_correction_mask)
    controlled_rmse_before = float(np.sqrt(mean_squared_error(log_values, contaminated)))
    controlled_rmse_after = float(np.sqrt(mean_squared_error(log_values, controlled_cleaned)))

    polynomial_results = []
    polynomial_fits: dict[int, np.ndarray] = {}
    for degree in (1, 2, 3):
        result, fitted = evaluate_polynomial(cleaned_log, degree)
        polynomial_results.append(result)
        polynomial_fits[degree] = fitted
    selected_polynomial = min(polynomial_results, key=lambda item: (item.cv_rmse_log, item.degree))

    phantom_matches = np.flatnonzero(data["date"].to_numpy() == PHANTOM_DATE.to_datetime64())
    if not phantom_matches.size:
        raise ValueError("Phantom Liberty event month is absent from the dataset")
    phantom_index = int(phantom_matches[0])
    nonlinear_result, nonlinear_fit = evaluate_event_model(cleaned_log, phantom_index)

    metric_rows = []
    for result in polynomial_results:
        row = result.to_dict()
        row["model"] = f"Поліном {result.degree}"
        metric_rows.append(row)
    nonlinear_row = nonlinear_result.to_dict()
    nonlinear_row["model"] = "Подієва R&D"
    metric_rows.append(nonlinear_row)
    model_metrics = pd.DataFrame(metric_rows)

    selected_coefficients = np.asarray(selected_polynomial.coefficients)
    polynomial_scale = float(len(cleaned_log) - 1)
    nonlinear_parameters = np.asarray(nonlinear_result.parameters)
    nonlinear_scale = float(len(cleaned_log) - 1)
    forecast_bundle = build_forecasts(
        dates=data["date"],
        observed_log=cleaned_log,
        polynomial_degree=selected_polynomial.degree,
        polynomial_coefficients=selected_coefficients,
        polynomial_scale=polynomial_scale,
        polynomial_fitted=polynomial_fits[selected_polynomial.degree],
        nonlinear_parameters=nonlinear_parameters,
        nonlinear_scale=nonlinear_scale,
        nonlinear_fitted=nonlinear_fit,
        phantom_index=phantom_index,
        seed=RANDOM_SEED,
    )
    backtest = _backtest(cleaned_log, (1, 2, 3), phantom_index)
    model_comparison = model_metrics.merge(backtest, on="model", how="left", validate="one_to_one")

    data.to_csv(results_dir / "cleaned_monthly_reviews.csv", index=False, date_format="%Y-%m-%d")
    anomalies.to_csv(results_dir / "anomaly_detection.csv", index=False)
    sensitivity.to_csv(results_dir / "anomaly_threshold_sensitivity.csv", index=False)
    model_comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    backtest.to_csv(results_dir / "backtest_last_12_months.csv", index=False)
    forecast_bundle.frame.to_csv(results_dir / "forecast_35_months.csv", index=False, date_format="%Y-%m-%d")
    injected_detection["original_log"] = log_values
    injected_detection["contaminated_log"] = contaminated
    injected_detection["cleaned_log"] = controlled_cleaned
    injected_detection["injected_anomaly"] = injected_detection["index"].isin(
        detector_validation.injected_indices
    )
    injected_detection["pchip_corrected"] = controlled_correction_mask
    injected_detection.insert(1, "date", data["date"].dt.strftime("%Y-%m-%d"))
    injected_detection.to_csv(results_dir / "injected_anomaly_validation.csv", index=False)

    summary: dict[str, Any] = {
        "random_seed": RANDOM_SEED,
        "data_quality": asdict(quality),
        "anomalies": {
            "method": "Hampel filter on polynomial-trend residuals",
            "window": 5,
            "threshold": selected_threshold,
            "threshold_selection": "maximum F1 on controlled anomalies",
            "threshold_candidates": list(threshold_candidates),
            "statistical_count": int(anomalies["statistical_anomaly"].sum()),
            "protected_count": int((anomalies["statistical_anomaly"] & anomalies["protected_event"]).sum()),
            "corrected_count": int(anomalies["correction_candidate"].sum()),
            "protected_months": sorted(PROTECTED_EVENT_MONTHS),
            "domain_verified_months": anomalies.loc[
                anomalies["statistical_anomaly"], "date"
            ].tolist(),
            "decision_rule": (
                "Statistical peaks are retained when the frozen source is internally consistent; "
                "PCHIP correction is applied only to unverified measurement errors."
            ),
        },
        "detector_validation": asdict(detector_validation),
        "controlled_cleaning": {
            "method": "PCHIP interpolation",
            "corrected_count": int(controlled_correction_mask.sum()),
            "rmse_before_log": controlled_rmse_before,
            "rmse_after_log": controlled_rmse_after,
        },
        "models": model_comparison.to_dict(orient="records"),
        "polynomial_models": [item.to_dict() for item in polynomial_results],
        "selected_polynomial_degree": selected_polynomial.degree,
        "nonlinear_model": nonlinear_result.to_dict(),
        "nonlinear_parameter_names": [
            "beta0", "beta1", "release_amplitude", "release_decay",
            "phantom_amplitude", "phantom_decay",
        ],
        "phantom_event_date": PHANTOM_DATE.date().isoformat(),
        "forecast_horizon": forecast_bundle.horizon,
        "forecast_start": forecast_bundle.frame["date"].min().date().isoformat(),
        "forecast_end": forecast_bundle.frame["date"].max().date().isoformat(),
        "backtest": backtest.to_dict(orient="records"),
    }
    (results_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    create_figures(
        data=data,
        anomaly_table=anomalies,
        cleaned_log=cleaned_log,
        polynomial_fits=polynomial_fits,
        nonlinear_fit=nonlinear_fit,
        model_metrics=model_metrics,
        forecast=forecast_bundle.frame,
        contaminated_log=contaminated,
        controlled_cleaned_log=controlled_cleaned,
        injected_indices=detector_validation.injected_indices,
        detected_indices=detector_validation.detected_indices,
        threshold_sensitivity=sensitivity,
        selected_threshold=selected_threshold,
        output_dir=figures_dir,
    )
    _print_summary(summary)
    print("\nАналіз завершено. Результати збережено у results/.")
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    run_analysis(root / "data", root / "results")
