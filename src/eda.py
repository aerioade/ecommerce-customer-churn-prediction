"""Exploratory Data Analysis (EDA), duplicate audit, and IQR outlier detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    CONTINUOUS_NUMERIC_FEATURES,
    EDA_SUMMARY_PATH,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.data_loader import load_dataset, standardize_categories


def compute_target_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Compute class counts, percentages, and imbalance ratio for the target column."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    total = int(len(df))
    non_churn = int(counts.get(0, 0))
    churn = int(counts.get(1, 0))
    imbalance_ratio = round(non_churn / churn, 4) if churn > 0 else None

    return {
        "total_rows": total,
        "non_churn_count": non_churn,
        "non_churn_pct": round((non_churn / total) * 100, 2) if total else 0.0,
        "churn_count": churn,
        "churn_pct": round((churn / total) * 100, 2) if total else 0.0,
        "imbalance_ratio_neg_to_pos": imbalance_ratio,
    }


def compute_missing_values(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute missing value counts and percentages for every column."""
    total = int(len(df))
    missing_summary: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_summary[col] = {
            "missing_count": missing_count,
            "missing_pct": round((missing_count / total) * 100, 2) if total else 0.0,
        }
    return missing_summary


def compute_churn_rate_by_category(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Compute churn counts and churn rates per category for categorical features."""
    result: dict[str, list[dict[str, Any]]] = {}
    for col in CATEGORICAL_FEATURES:
        grouped = (
            df.groupby(col, dropna=False)[TARGET_COLUMN]
            .agg(total_customers="count", churn_count="sum", churn_rate="mean")
            .reset_index()
            .sort_values(by="churn_rate", ascending=False)
        )
        rows = []
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "category": str(row[col]),
                    "total_customers": int(row["total_customers"]),
                    "churn_count": int(row["churn_count"]),
                    "non_churn_count": int(row["total_customers"] - row["churn_count"]),
                    "churn_rate_pct": round(float(row["churn_rate"]) * 100, 2),
                }
            )
        result[col] = rows
    return result


def detect_outliers_iqr(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect outliers using the IQR method (documented only, never removed).

    Formula:
        IQR = Q3 - Q1
        Lower Bound = Q1 - 1.5 * IQR
        Upper Bound = Q3 + 1.5 * IQR
    """
    outlier_report: list[dict[str, Any]] = []
    for col in CONTINUOUS_NUMERIC_FEATURES:
        series = df[col].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        below_mask = series < lower_bound
        above_mask = series > upper_bound
        outlier_count = int((below_mask | above_mask).sum())
        valid_count = int(len(series))

        outlier_report.append(
            {
                "feature": col,
                "valid_count": valid_count,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "min_value": round(float(series.min()), 4),
                "max_value": round(float(series.max()), 4),
                "outliers_below_lower": int(below_mask.sum()),
                "outliers_above_upper": int(above_mask.sum()),
                "outlier_count": outlier_count,
                "outlier_pct": round((outlier_count / valid_count) * 100, 2) if valid_count else 0.0,
                "action": "documented_only_not_removed",
            }
        )
    return outlier_report


def compute_numeric_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute descriptive statistics for numeric predictor features."""
    summary: dict[str, dict[str, float]] = {}
    for col in NUMERIC_FEATURES:
        series = df[col].dropna()
        summary[col] = {
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
        }
    return summary


def run_eda(
    raw_data_path: Path = RAW_DATA_PATH,
    output_path: Path = EDA_SUMMARY_PATH,
) -> dict[str, Any]:
    """Execute full EDA on raw and deduplicated datasets and save JSON summary.

    Args:
        raw_data_path: Path to the raw CSV dataset.
        output_path: Destination path for `eda_summary.json`.

    Returns:
        Dictionary containing the complete EDA summary.
    """
    # Load raw dataset without standardization first to audit exact raw state
    df_raw = load_dataset(raw_data_path, require_target=True, standardize=False)
    raw_duplicates = int(df_raw.duplicated().sum())
    raw_unique_rows = int(len(df_raw) - raw_duplicates)

    # Apply deterministic category standardization then deduplicate (anti-leakage step)
    df_standardized = standardize_categories(df_raw)
    standardized_duplicates = int(df_standardized.duplicated().sum())
    df_clean = df_standardized.drop_duplicates().reset_index(drop=True)

    eda_summary: dict[str, Any] = {
        "dataset_overview": {
            "source_file": str(raw_data_path.name),
            "raw_rows": int(df_raw.shape[0]),
            "raw_columns": int(df_raw.shape[1]),
            "raw_duplicate_rows": raw_duplicates,
            "raw_unique_rows_before_standardization": raw_unique_rows,
            "duplicates_after_category_standardization": standardized_duplicates,
            "clean_unique_rows": int(df_clean.shape[0]),
            "category_standardization_note": (
                "Mapped 'Mobile' -> 'Mobile Phone' on PreferedOrderCat prior to deduplication."
            ),
        },
        "target_distribution": {
            "raw": compute_target_distribution(df_raw),
            "clean": compute_target_distribution(df_clean),
        },
        "missing_values": {
            "raw": compute_missing_values(df_raw),
            "clean": compute_missing_values(df_clean),
        },
        "churn_rate_by_category": compute_churn_rate_by_category(df_clean),
        "iqr_outlier_detection": detect_outliers_iqr(df_clean),
        "numeric_feature_summary_clean": compute_numeric_summary(df_clean),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    return eda_summary


if __name__ == "__main__":
    summary = run_eda()
    overview = summary["dataset_overview"]
    print(f"EDA summary saved to: {EDA_SUMMARY_PATH}")
    print(
        f"Raw: {overview['raw_rows']} rows x {overview['raw_columns']} cols | "
        f"Raw duplicates: {overview['raw_duplicate_rows']} | "
        f"Duplicates after standardization: {overview['duplicates_after_category_standardization']} | "
        f"Clean unique rows: {overview['clean_unique_rows']}"
    )
