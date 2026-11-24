"""Tune, evaluate, and save the three churn classification pipelines."""

import json

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    EVALUATION_METRICS_PATH,
    LR_PIPELINE_PATH,
    N_SPLITS,
    RANDOM_STATE,
    RF_PIPELINE_PATH,
    XGB_PIPELINE_PATH,
)
from src.preprocessing import (
    build_preprocessor,
    compute_scale_pos_weight,
    prepare_and_split_data,
)


def build_searches(y_train):
    """Build training-only searches with a fresh preprocessor in each pipeline."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    options = {"cv": cv, "scoring": "f1", "error_score": "raise"}
    random_options = {**options, "n_iter": 12, "random_state": RANDOM_STATE}

    logistic = Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=True)),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
        )),
    ])
    forest = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    xgboost = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(
            scale_pos_weight=compute_scale_pos_weight(y_train),
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])

    return {
        "logistic_regression": (LR_PIPELINE_PATH, GridSearchCV(logistic, {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs"],
        }, **options)),
        "random_forest": (RF_PIPELINE_PATH, RandomizedSearchCV(forest, {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        }, **random_options)),
        "xgboost": (XGB_PIPELINE_PATH, RandomizedSearchCV(xgboost, {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5, 7, 9],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        }, **random_options)),
    }


def evaluate_pipeline(pipeline, X_test, y_test):
    """Compute holdout classification metrics and ROC coordinates."""
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    fpr, tpr, thresholds = roc_curve(y_test, probabilities, pos_label=1)

    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": matrix.tolist(),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "specificity": tn / (tn + fp),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "fpr": fp / (fp + tn),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            # The initial infinite threshold represents predicting no positives.
            "thresholds": [float(t) if np.isfinite(t) else None for t in thresholds],
        },
    }


def train_models():
    """Fit all candidates, select by holdout F1 and AUC, and save the results."""
    X_train, X_test, y_train, y_test = prepare_and_split_data(save_files=False)
    fitted = {}
    metrics = {}
    for name, (path, search) in build_searches(y_train).items():
        search.fit(X_train, y_train)
        fitted[name] = (path, search.best_estimator_)
        metrics[name] = evaluate_pipeline(search.best_estimator_, X_test, y_test)
        metrics[name].update(best_params=search.best_params_, cv_f1=float(search.best_score_))
        print(f"{name}: test F1={metrics[name]['f1']:.4f}, "
              f"ROC-AUC={metrics[name]['roc_auc']:.4f}", flush=True)

    best_model = max(metrics, key=lambda name: (metrics[name]["f1"], metrics[name]["roc_auc"]))
    result = {"best_model": best_model, "models": metrics}
    for path, pipeline in [*fitted.values(), (BEST_MODEL_PATH, fitted[best_model][1])]:
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(pipeline, path, compress=3)

    EVALUATION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_METRICS_PATH.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Best model: {best_model}")
    return result


if __name__ == "__main__":
    train_models()
