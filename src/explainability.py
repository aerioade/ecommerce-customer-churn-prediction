"""Generate global and local SHAP explanations for the selected churn model."""

import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    FEATURE_IMPORTANCE_PATH,
    RANDOM_STATE,
    SHAP_BACKGROUND_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from src.data_loader import load_dataset, standardize_categories, validate_schema
from src.preprocessing import get_feature_names


class ChurnExplainer:
    """Reuse the winning pipeline's fitted preprocessing and SHAP explainer."""

    def __init__(self, pipeline, background):
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]
        self.feature_names = get_feature_names(self.preprocessor)
        self.positive_index = list(self.classifier.classes_).index(1)
        if isinstance(self.classifier, LogisticRegression):
            self.explainer = shap.LinearExplainer(self.classifier, background)
            self.output_scale = "log_odds"
        elif isinstance(self.classifier, (RandomForestClassifier, XGBClassifier)):
            self.explainer = shap.TreeExplainer(self.classifier)
            self.output_scale = (
                "probability" if isinstance(self.classifier, RandomForestClassifier)
                else "log_odds"
            )
        else:
            raise TypeError(f"Unsupported classifier: {type(self.classifier).__name__}")

    def explain(self, observations: pd.DataFrame) -> shap.Explanation:
        """Explain the positive class with feature names in transformed order."""
        validate_schema(observations, require_target=False)
        transformed = self.preprocessor.transform(standardize_categories(observations))
        explanation = self.explainer(transformed)
        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)
        if values.ndim == 3:
            values = values[:, :, self.positive_index]
            base_values = base_values[..., self.positive_index]
        if values.shape != transformed.shape:
            raise ValueError(f"Unexpected SHAP values shape: {values.shape}")
        return shap.Explanation(
            values=values,
            base_values=np.broadcast_to(base_values, (len(transformed),)),
            data=transformed,
            feature_names=self.feature_names,
        )

    def explain_local(self, observation: pd.DataFrame) -> dict:
        """Return a waterfall-ready explanation and up to three factors per direction."""
        if len(observation) != 1:
            raise ValueError("Local explanation requires exactly one observation.")
        explanation = self.explain(observation)[0]
        ranked = _rank_features(self.feature_names, explanation.values)
        return {
            "explanation": explanation,
            "drivers": [item for item in ranked if item["importance"] > 0][:3],
            "protectors": [item for item in reversed(ranked) if item["importance"] < 0][:3],
            "output_scale": self.output_scale,
        }


def _rank_features(names, scores):
    """List feature scores in descending order."""
    return [
        {"feature": name, "importance": float(score)}
        for name, score in sorted(zip(names, scores), key=lambda pair: -pair[1])
    ]


def load_explainer() -> ChurnExplainer:
    """Load the winning pipeline and its matching SHAP background artifact."""
    pipeline = joblib.load(BEST_MODEL_PATH)
    metadata = joblib.load(SHAP_BACKGROUND_PATH)
    explainer = ChurnExplainer(pipeline, metadata["background"])
    if (metadata["feature_names"] != explainer.feature_names
            or metadata["classifier"] != type(explainer.classifier).__name__):
        raise ValueError("SHAP metadata does not match the pipeline. Rerun explainability.")
    return explainer


def generate_shap_artifacts():
    """Compare test-set SHAP importance with the selected model's native importance."""
    pipeline = joblib.load(BEST_MODEL_PATH)
    train = load_dataset(TRAIN_DATA_PATH)
    test = load_dataset(TEST_DATA_PATH)
    background = pipeline.named_steps["preprocessor"].transform(
        train.sample(n=min(len(train), 100), random_state=RANDOM_STATE)
    )
    explainer = ChurnExplainer(pipeline, background)
    global_scores = np.abs(explainer.explain(test).values).mean(axis=0)
    classifier = explainer.classifier
    native_scores = getattr(classifier, "feature_importances_", None)
    if isinstance(classifier, XGBClassifier):
        native_method = classifier.importance_type or "gain"
    elif isinstance(classifier, RandomForestClassifier):
        native_method = classifier.criterion
    else:
        native_method = None

    result = {
        "classifier": type(classifier).__name__,
        "output_scale": explainer.output_scale,
        "test_rows": len(test),
        "shap_importance": _rank_features(explainer.feature_names, global_scores),
        "model_importance_method": native_method,
        "model_importance": (
            _rank_features(explainer.feature_names, native_scores)
            if native_scores is not None else None
        ),
    }
    joblib.dump({
        "classifier": result["classifier"],
        "feature_names": explainer.feature_names,
        "output_scale": explainer.output_scale,
        "background": background,
    }, SHAP_BACKGROUND_PATH, compress=3)
    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_IMPORTANCE_PATH.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"SHAP artifacts generated for {result['classifier']} on {len(test)} test rows.")


if __name__ == "__main__":
    generate_shap_artifacts()
