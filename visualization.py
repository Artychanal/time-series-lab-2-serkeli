from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#1b8fc5"
ORANGE = "#f2a65a"
RED = "#d65a4a"
GREEN = "#4a9d64"
DARK = "#171a21"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, facecolor="white")
    plt.close()


def create_figures(
    data: pd.DataFrame,
    anomaly_table: pd.DataFrame,
    cleaned_log: np.ndarray,
    polynomial_fits: dict[int, np.ndarray],
    nonlinear_fit: np.ndarray,
    model_metrics: pd.DataFrame,
    forecast: pd.DataFrame,
    contaminated_log: np.ndarray,
    controlled_cleaned_log: np.ndarray,
    injected_indices: tuple[int, ...],
    detected_indices: tuple[int, ...],
    threshold_sensitivity: pd.DataFrame,
    selected_threshold: float,
    output_dir: Path,
) -> None:
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = data["date"]
    values = data["total_reviews"].to_numpy(float)

    plt.figure(figsize=(11, 5.2))
    plt.plot(dates, values, color=BLUE, linewidth=2, label="Місячні відгуки")
    plt.yscale("log")
    plt.title("Місячна кількість відгуків Cyberpunk 2077 у Steam")
    plt.xlabel("Місяць")
    plt.ylabel("Кількість відгуків, log-шкала")
    plt.legend()
    _save(output_dir / "01_input_series.png")

    plt.figure(figsize=(11, 5.2))
    plt.plot(dates, data["log_total_reviews"], color=DARK, linewidth=1.6, label="Вхідний ряд")
    flagged = anomaly_table["statistical_anomaly"].to_numpy(bool)
    protected = anomaly_table["protected_event"].to_numpy(bool) & flagged
    candidates = anomaly_table["correction_candidate"].to_numpy(bool)
    plt.scatter(dates[protected], data.loc[protected, "log_total_reviews"], color=ORANGE, s=60,
                label="Подієві піки (збережено)", zorder=3)
    plt.scatter(dates[candidates], data.loc[candidates, "log_total_reviews"], color=RED, marker="x", s=70,
                label="Кандидати на очищення", zorder=3)
    plt.plot(dates, cleaned_log, color=GREEN, linestyle="--", linewidth=1.3, label="Очищений ряд")
    plt.title("Hampel-аналіз аномалій на залишках тренду")
    plt.xlabel("Місяць")
    plt.ylabel("log(1 + кількість відгуків)")
    plt.legend()
    _save(output_dir / "02_anomaly_detection.png")

    plt.figure(figsize=(11, 5.2))
    plt.plot(dates, cleaned_log, color=DARK, linewidth=1.7, label="Дані")
    for degree, fitted in polynomial_fits.items():
        plt.plot(dates, fitted, linewidth=1.4, label=f"Поліном {degree}-го степеня")
    plt.plot(dates, nonlinear_fit, color=RED, linewidth=2.0, label="Подієва R&D-модель")
    plt.title("Порівняння навчених моделей")
    plt.xlabel("Місяць")
    plt.ylabel("log(1 + кількість відгуків)")
    plt.legend()
    _save(output_dir / "03_model_fits.png")

    plt.figure(figsize=(9, 4.8))
    order = model_metrics.sort_values("cv_rmse_log")
    plt.bar(order["model"], order["cv_rmse_log"], color=[RED if "R&D" in name else BLUE for name in order["model"]])
    plt.ylabel("Expanding-window CV RMSE, log-шкала")
    plt.title("Якість моделей на часовій валідації")
    plt.xticks(rotation=15, ha="right")
    _save(output_dir / "04_model_validation.png")

    plt.figure(figsize=(12, 5.5))
    plt.plot(dates, values, color=DARK, linewidth=1.8, label="Фактичні дані")
    plt.plot(forecast["date"], forecast["polynomial_forecast"], color=BLUE, linewidth=1.8,
             label="Найкращий поліном")
    plt.fill_between(forecast["date"], forecast["polynomial_lower_95"], forecast["polynomial_upper_95"],
                     color=BLUE, alpha=0.14)
    plt.plot(forecast["date"], forecast["nonlinear_forecast"], color=RED, linewidth=2.0,
             label="Подієва R&D-модель")
    plt.fill_between(forecast["date"], forecast["nonlinear_lower_95"], forecast["nonlinear_upper_95"],
                     color=RED, alpha=0.14)
    plt.yscale("log")
    plt.title("Прогноз на 0,5 інтервалу спостережень із 95% інтервалами")
    plt.xlabel("Місяць")
    plt.ylabel("Кількість відгуків, log-шкала")
    plt.legend()
    _save(output_dir / "05_forecast.png")

    plt.figure(figsize=(11, 5.2))
    plt.plot(dates, contaminated_log, color=DARK, linewidth=1.2, label="Ряд із аномаліями")
    plt.plot(dates, data["log_total_reviews"], color=BLUE, linestyle="--", linewidth=1.3,
             label="Початковий ряд")
    plt.plot(dates, controlled_cleaned_log, color=GREEN, linewidth=1.8,
             label="Ряд після PCHIP-очищення")
    injected = np.array(injected_indices, dtype=int)
    detected = np.array(detected_indices, dtype=int)
    plt.scatter(dates.iloc[injected], contaminated_log[injected], color=RED, marker="x", s=80,
                label="Внесені аномалії", zorder=4)
    if detected.size:
        plt.scatter(dates.iloc[detected], contaminated_log[detected], facecolors="none", edgecolors=BLUE,
                    s=110, linewidths=1.8, label="Виявлено алгоритмом", zorder=3)
    plt.title("Виявлення та очищення контрольованих аномалій")
    plt.xlabel("Місяць")
    plt.ylabel("log(1 + кількість відгуків)")
    plt.legend()
    _save(output_dir / "06_detector_verification.png")

    plt.figure(figsize=(9, 4.8))
    plt.plot(threshold_sensitivity["threshold"], threshold_sensitivity["precision"],
             marker="o", linewidth=1.7, label="Precision")
    plt.plot(threshold_sensitivity["threshold"], threshold_sensitivity["recall"],
             marker="o", linewidth=1.7, label="Recall")
    plt.plot(threshold_sensitivity["threshold"], threshold_sensitivity["f1"],
             marker="o", linewidth=2.2, color=RED, label="F1")
    plt.axvline(selected_threshold, color=DARK, linestyle="--", linewidth=1.3,
                label=f"Обраний поріг = {selected_threshold:.1f}")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Поріг Hampel-фільтра")
    plt.ylabel("Значення метрики")
    plt.title("Оптимізація порога виявлення аномалій")
    plt.legend()
    _save(output_dir / "07_threshold_optimization.png")
