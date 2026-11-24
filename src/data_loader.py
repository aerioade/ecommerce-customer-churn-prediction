"""Dataset loading, schema validation, and deterministic category standardization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    ALL_COLUMNS,
    CATEGORY_MAPPING,
    FEATURE_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """Validate that required columns exist in the DataFrame.

    Args:
        df: Input DataFrame to validate.
        require_target: If True, requires both the 10 predictor columns and
            the target column (`Churn`). If False, requires only the 10
            predictor columns (e.g., for batch inference).

    Raises:
        ValueError: If one or more required columns are missing.
    """
    required_columns = ALL_COLUMNS if require_target else FEATURE_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan pada dataset: {missing_columns}. "
            f"Kolom yang dibutuhkan: {required_columns}"
        )


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic category label standardization without fitting on data.

    Maps `'Mobile'` -> `'Mobile Phone'` in `PreferedOrderCat` as defined in
    `src.config.CATEGORY_MAPPING`. Safe to execute before train-test split.

    Args:
        df: Input DataFrame.

    Returns:
        A copy of the DataFrame with standardized category values.
    """
    standardized = df.copy()
    for col, mapping in CATEGORY_MAPPING.items():
        if col in standardized.columns:
            standardized[col] = standardized[col].replace(mapping)
    return standardized


def _read_tabular_source(source: str | Path | Any, filename: str | None = None) -> pd.DataFrame:
    """Read a CSV or Excel file from a path or file-like object."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File dataset tidak ditemukan: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(
            f"Format file '{suffix}' tidak didukung. Gunakan file .csv atau .xlsx."
        )

    # Support file-like objects (e.g., Streamlit UploadedFile, BytesIO)
    inferred_name = filename or getattr(source, "name", "")
    suffix = Path(inferred_name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    raise ValueError(
        f"Format file '{suffix or 'unknown'}' tidak didukung. Gunakan file .csv atau .xlsx."
    )


def load_dataset(
    source: str | Path | Any = RAW_DATA_PATH,
    require_target: bool = True,
    standardize: bool = True,
    filename: str | None = None,
) -> pd.DataFrame:
    """Load dataset from CSV/Excel, validate schema, and optionally standardize labels.

    Args:
        source: File path (`str`/`Path`) or file-like object (`.csv` or `.xlsx`).
            Defaults to `RAW_DATA_PATH`.
        require_target: Whether to require the `Churn` target column.
        standardize: Whether to apply deterministic category standardization.
        filename: Optional filename hint when `source` is a file-like object
            without a `.name` attribute.

    Returns:
        Validated DataFrame containing the expected columns.
    """
    df = _read_tabular_source(source, filename=filename)
    validate_schema(df, require_target=require_target)
    if standardize:
        df = standardize_categories(df)
    return df
