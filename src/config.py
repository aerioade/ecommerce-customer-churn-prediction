"""Global configuration, paths, schema constants, and thresholds."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
APP_DIR = BASE_DIR / "app"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Data file paths
RAW_DATA_PATH = RAW_DATA_DIR / "data_ecommerce_customer_churn.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.csv"

# Model artifact paths
LR_PIPELINE_PATH = MODELS_DIR / "logistic_regression_pipeline.joblib"
RF_PIPELINE_PATH = MODELS_DIR / "random_forest_pipeline.joblib"
XGB_PIPELINE_PATH = MODELS_DIR / "xgboost_pipeline.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model_pipeline.joblib"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.joblib"

# Evaluation & analysis artifact paths
EVALUATION_METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"
EDA_SUMMARY_PATH = ARTIFACTS_DIR / "eda_summary.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.json"

# Reproducibility & CV settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# Dataset schema (10 predictors + 1 target)
NUMERIC_FEATURES = [
    "Tenure",
    "WarehouseToHome",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "DaySinceLastOrder",
    "CashbackAmount",
]

CATEGORICAL_FEATURES = [
    "PreferedOrderCat",
    "MaritalStatus",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Churn"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Continuous numeric features for IQR outlier documentation
CONTINUOUS_NUMERIC_FEATURES = [
    "Tenure",
    "WarehouseToHome",
    "NumberOfDeviceRegistered",
    "NumberOfAddress",
    "DaySinceLastOrder",
    "CashbackAmount",
]

# Deterministic category standardization
CATEGORY_MAPPING = {
    "PreferedOrderCat": {
        "Mobile": "Mobile Phone",
    }
}

# 3-Tier Risk Stratification thresholds
HIGH_RISK_THRESHOLD = 0.65
MEDIUM_RISK_THRESHOLD = 0.35
