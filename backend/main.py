"""FastAPI service that exposes the existing churn artifacts to the web UI."""

from __future__ import annotations

import io
import json
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import (
    BEST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    EDA_SUMMARY_PATH,
    EVALUATION_METRICS_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
)
from src.data_loader import load_dataset, standardize_categories, validate_schema
from src.explainability import load_explainer
from src.risk_engine import apply_risk_to_batch, classify_risk

app = FastAPI(title="ChurnGuard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerFeatures(BaseModel):
    Tenure: float | None = None
    WarehouseToHome: float | None = None
    NumberOfDeviceRegistered: int
    PreferedOrderCat: str
    SatisfactionScore: int
    MaritalStatus: str
    NumberOfAddress: int
    Complain: int
    DaySinceLastOrder: float | None = None
    CashbackAmount: float


def _load_json(path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records_json_safe(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame rows to JSON-compliant dicts with NaN/NA replaced by None."""
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def _pipeline():
    return joblib.load(BEST_MODEL_PATH)


def _positive_probabilities(frame: pd.DataFrame):
    pipeline = _pipeline()
    positive_index = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(frame[FEATURE_COLUMNS])[:, positive_index]


def _local_explanation(frame: pd.DataFrame) -> dict[str, Any]:
    details = load_explainer().explain_local(frame)
    return {
        "output_scale": details["output_scale"],
        "drivers": details["drivers"],
        "protectors": details["protectors"],
    }


OPTIONAL_ID_COLUMNS = {"customerid", "customer_id"}


def _find_optional_id_column(frame: pd.DataFrame) -> str | None:
    """Return the original column name if an optional CustomerID column is present."""
    for column in frame.columns:
        if str(column).strip().lower() in OPTIONAL_ID_COLUMNS:
            return str(column)
    return None


def _validate_inference_frame(frame: pd.DataFrame) -> list[str]:
    """Validate batch inference inputs without changing the training pipeline."""
    validate_schema(frame, require_target=False)
    warnings: list[str] = []
    id_column = _find_optional_id_column(frame)
    ignored = [
        column for column in frame.columns
        if column not in FEATURE_COLUMNS and column != id_column
    ]
    if ignored:
        warnings.append(f"Kolom tambahan diabaikan: {', '.join(ignored)}.")

    allowed_categories = {
        "PreferedOrderCat": {"Mobile Phone", "Fashion", "Grocery", "Laptop & Accessory", "Others"},
        "MaritalStatus": {"Single", "Married", "Divorced"},
    }
    numeric_values: dict[str, pd.Series] = {}
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce")
        numeric_values[column] = values
        if values[frame[column].notna()].isna().any():
            raise ValueError(f"Kolom {column} harus berisi angka atau dikosongkan jika termasuk fitur yang boleh missing.")
        missing = int(values.isna().sum())
        if column not in {"Tenure", "WarehouseToHome", "DaySinceLastOrder"} and missing:
            raise ValueError(f"Kolom {column} memiliki {missing} nilai kosong. Isi semua nilai ini sebelum upload.")
        if (values.dropna() < 0).any():
            raise ValueError(f"Kolom {column} tidak boleh berisi angka negatif.")
    if not numeric_values["NumberOfDeviceRegistered"].dropna().between(1, 6).all():
        raise ValueError("NumberOfDeviceRegistered harus berada pada rentang 1 sampai 6.")
    if not numeric_values["SatisfactionScore"].dropna().between(1, 5).all():
        raise ValueError("SatisfactionScore harus berada pada rentang 1 sampai 5.")
    if not numeric_values["NumberOfAddress"].dropna().ge(1).all():
        raise ValueError("NumberOfAddress harus bernilai minimal 1.")
    if not numeric_values["Complain"].dropna().isin([0, 1]).all():
        raise ValueError("Complain hanya boleh bernilai 0 atau 1.")
    for column, allowed in allowed_categories.items():
        if frame[column].isna().any():
            raise ValueError(f"Kolom {column} tidak boleh kosong.")
        invalid = sorted(set(frame[column].dropna().astype(str)) - allowed)
        if invalid:
            raise ValueError(f"Nilai {column} tidak dikenali: {', '.join(invalid)}.")
    return warnings


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "XGBoost"}


@app.get("/api/model-analytics")
def model_analytics() -> dict[str, Any]:
    return _load_json(EVALUATION_METRICS_PATH)


@app.get("/api/eda-insights")
def eda_insights() -> dict[str, Any]:
    return _load_json(EDA_SUMMARY_PATH)


@app.get("/api/feature-importance")
def feature_importance() -> dict[str, Any]:
    return _load_json(FEATURE_IMPORTANCE_PATH)


@app.get("/api/dashboard-summary")
def dashboard_summary() -> dict[str, Any]:
    """Predict the repository dataset to supply the operational overview."""
    dataset = load_dataset(RAW_DATA_PATH, require_target=True, standardize=True)
    customers = dataset.drop_duplicates().reset_index(drop=True)[FEATURE_COLUMNS].copy()
    probabilities = _positive_probabilities(customers)
    results = apply_risk_to_batch(customers, probabilities)
    results.insert(0, "Customer_Index", range(1, len(results) + 1))
    counts = results["Risk_Tier"].value_counts().to_dict()
    priority = results.sort_values("Churn_Probability", ascending=False).head(7)
    return {
        "summary": {
            "total_customers": int(len(results)),
            "high_risk": int(counts.get("High Risk", 0)),
            "medium_risk": int(counts.get("Medium Risk", 0)),
            "low_risk": int(counts.get("Low Risk", 0)),
            "average_probability": float(results["Churn_Probability"].mean()),
        },
        "priority_customers": _records_json_safe(priority),
    }


@app.post("/api/predict/individual")
def predict_individual(customer: CustomerFeatures) -> dict[str, Any]:
    frame = standardize_categories(pd.DataFrame([customer.model_dump()]))
    try:
        _validate_inference_frame(frame)
        probability = float(_positive_probabilities(frame)[0])
        return {
            "probability": probability,
            "prediction": int(probability >= 0.50),
            "risk": classify_risk(probability),
            "explanation": _local_explanation(frame),
        }
    except Exception as exc:  # Convert ML input failures into an API-safe response.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _read_upload(upload: UploadFile, content: bytes) -> pd.DataFrame:
    name = upload.filename or "upload"
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    source = io.BytesIO(content)
    if suffix == "csv":
        return pd.read_csv(source)
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(source)
    raise ValueError("Format file tidak didukung. Gunakan CSV atau XLSX.")


@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        frame = _read_upload(file, await file.read())
        standardized = standardize_categories(frame)
        warnings = _validate_inference_frame(standardized)
        id_column = _find_optional_id_column(standardized)
        customers = standardized[FEATURE_COLUMNS].copy()
        probabilities = _positive_probabilities(customers)
        results = apply_risk_to_batch(customers, probabilities)
        if id_column is not None:
            results.insert(0, "CustomerID", standardized[id_column].values)
        counts = results["Risk_Tier"].value_counts().to_dict()
        return {
            "summary": {
                "total_customers": int(len(results)),
                "high_risk": int(counts.get("High Risk", 0)),
                "medium_risk": int(counts.get("Medium Risk", 0)),
                "low_risk": int(counts.get("Low Risk", 0)),
                "average_probability": float(results["Churn_Probability"].mean()),
            },
            "columns": list(results.columns),
            "rows": _records_json_safe(results),
            "warnings": warnings,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
