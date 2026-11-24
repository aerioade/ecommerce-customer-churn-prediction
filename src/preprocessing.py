"""Anti-data-leakage preprocessing pipeline, stratified splitting, and class weighting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
)
from src.data_loader import load_dataset


def prepare_and_split_data(
    raw_data_path: Path = RAW_DATA_PATH,
    train_output_path: Path = TRAIN_DATA_PATH,
    test_output_path: Path = TEST_DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_files: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, standardize, deduplicate, and perform stratified 80:20 train-test split.

    Follows strict anti-data-leakage protocol:
    1. Standardize deterministic category labels ('Mobile' -> 'Mobile Phone').
    2. Remove exact duplicate rows (`drop_duplicates().reset_index(drop=True)`)
       BEFORE splitting so identical samples never leak across train and test sets.
    3. Separate 10 predictor features (X) and target column (y).
    4. Execute stratified train-test split (80% train, 20% test) with `random_state=42`.

    Args:
        raw_data_path: Path to raw CSV dataset.
        train_output_path: Destination CSV path for training split.
        test_output_path: Destination CSV path for testing split.
        test_size: Proportion of dataset allocated to test split (0.20).
        random_state: Random seed for reproducibility (42).
        save_files: Whether to save `train_data.csv` and `test_data.csv`.

    Returns:
        Tuple of `(X_train, X_test, y_train, y_test)`.
    """
    df = load_dataset(raw_data_path, require_target=True, standardize=True)
    df_clean = df.drop_duplicates().reset_index(drop=True)

    X = df_clean[FEATURE_COLUMNS].copy()
    y = df_clean[TARGET_COLUMN].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    if save_files:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_output_path.parent.mkdir(parents=True, exist_ok=True)
        test_output_path.parent.mkdir(parents=True, exist_ok=True)

        train_df = X_train.copy()
        train_df[TARGET_COLUMN] = y_train
        train_df.to_csv(train_output_path, index=False)

        test_df = X_test.copy()
        test_df[TARGET_COLUMN] = y_test
        test_df.to_csv(test_output_path, index=False)

    return X_train, X_test, y_train, y_test


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Create a scikit-learn ColumnTransformer for the 10 predictor features.

    Args:
        scale_numeric: Set to True for Logistic Regression (`SimpleImputer(median)`
            + `StandardScaler()`). Set to False for tree-based models (Random Forest
            and XGBoost use `SimpleImputer(median)` only).

    Returns:
        Unfitted `ColumnTransformer` supporting `get_feature_names_out()`.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract clean transformed feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: A fitted `ColumnTransformer`.

    Returns:
        List of feature names produced after imputation and One-Hot Encoding.
    """
    return [str(name) for name in preprocessor.get_feature_names_out()]


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute XGBoost `scale_pos_weight` exclusively from `y_train` (no SMOTE).

    Formula:
        scale_pos_weight = N_negative_train / N_positive_train

    Args:
        y_train: Training target vector containing binary labels (0 and 1).

    Returns:
        Ratio of negative to positive samples in `y_train`.
    """
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    if pos_count == 0:
        raise ValueError("y_train tidak memiliki sampel kelas positif (Churn = 1).")
    return float(neg_count / pos_count)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_and_split_data()
    spw = compute_scale_pos_weight(y_train)

    lr_prep = build_preprocessor(scale_numeric=True)
    tree_prep = build_preprocessor(scale_numeric=False)

    X_train_lr = lr_prep.fit_transform(X_train)
    X_train_tree = tree_prep.fit_transform(X_train)
    feature_names = get_feature_names(tree_prep)

    print(f"Saved train split: {TRAIN_DATA_PATH} -> {X_train.shape[0]} rows")
    print(f"Saved test split:  {TEST_DATA_PATH} -> {X_test.shape[0]} rows")
    print(
        f"Train churn ratio: {y_train.mean():.4f} | "
        f"Test churn ratio: {y_test.mean():.4f} | "
        f"scale_pos_weight (train): {spw:.4f}"
    )
    print(
        f"Transformed shape (LR): {X_train_lr.shape} | "
        f"Transformed shape (Tree): {X_train_tree.shape}"
    )
    print(f"Encoded feature names ({len(feature_names)}): {feature_names}")
