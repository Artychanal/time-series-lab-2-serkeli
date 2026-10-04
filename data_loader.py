from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_END = pd.Timestamp("2026-08-01")


@dataclass(frozen=True)
class DataQualityReport:
    raw_observations: int
    observations: int
    period_start: str
    period_end: str
    excluded_months: tuple[str, ...]
    duplicate_months: int
    missing_values: int
    missing_calendar_months: tuple[str, ...]
    totals_consistent: bool


def load_and_validate_monthly_reviews(
    csv_path: Path, analysis_end: pd.Timestamp = ANALYSIS_END
) -> tuple[pd.DataFrame, DataQualityReport]:
    frame = pd.read_csv(csv_path, parse_dates=["date"])
    required = {
        "date", "positive_reviews", "negative_reviews", "total_reviews", "positive_share"
    }
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    raw_count = len(frame)
    frame = frame.sort_values("date").reset_index(drop=True)
    duplicates = int(frame["date"].duplicated().sum())
    missing_values = int(frame[list(required)].isna().sum().sum())
    totals_consistent = bool(
        (frame["total_reviews"] == frame["positive_reviews"] + frame["negative_reviews"]).all()
    )
    expected = pd.date_range(frame["date"].min(), frame["date"].max(), freq="MS")
    missing_months = expected.difference(frame["date"])

    if duplicates:
        raise ValueError(f"Dataset contains {duplicates} duplicate month(s)")
    if missing_values:
        raise ValueError(f"Dataset contains {missing_values} missing value(s)")
    if len(missing_months):
        raise ValueError(f"Missing calendar months: {missing_months.strftime('%Y-%m').tolist()}")
    if not totals_consistent:
        raise ValueError("total_reviews differs from positive_reviews + negative_reviews")
    if (frame[["positive_reviews", "negative_reviews", "total_reviews"]] < 0).any().any():
        raise ValueError("Review counts must be non-negative")

    selected = frame.loc[frame["date"] <= analysis_end].copy().reset_index(drop=True)
    excluded = frame.loc[frame["date"] > analysis_end, "date"]
    if selected.empty:
        raise ValueError("No observations remain inside the fixed analysis period")

    selected["t"] = np.arange(len(selected), dtype=float)
    selected["log_total_reviews"] = np.log1p(selected["total_reviews"].astype(float))
    report = DataQualityReport(
        raw_observations=raw_count,
        observations=len(selected),
        period_start=selected["date"].min().date().isoformat(),
        period_end=selected["date"].max().date().isoformat(),
        excluded_months=tuple(excluded.dt.strftime("%Y-%m")),
        duplicate_months=duplicates,
        missing_values=missing_values,
        missing_calendar_months=tuple(missing_months.strftime("%Y-%m")),
        totals_consistent=totals_consistent,
    )
    return selected, report
