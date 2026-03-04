from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

FEATURES = [
    "study_hours",
    "attendance_pct",
    "assignment_score",
    "midterm_score",
    "sleep_hours",
    "past_gpa",
    "expected_grade_factor",
]

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


class ModelService:
    def __init__(self):
        self.classifier = joblib.load(ARTIFACT_DIR / "classifier.joblib")
        self.baseline = joblib.load(ARTIFACT_DIR / "logistic_baseline.joblib")
        self.grade_model = joblib.load(ARTIFACT_DIR / "grade_regressor.joblib")
        self.background_data = joblib.load(ARTIFACT_DIR / "background_data.joblib")
        self._explainer = None

    @staticmethod
    def _risk_from_probability(prob_pass: float) -> str:
        if prob_pass >= 0.75:
            return "Low"
        if prob_pass >= 0.5:
            return "Medium"
        return "High"

    @staticmethod
    def _coherent_final_grade(prob_pass: float, raw_grade: float) -> float:
        # Prevent obvious contradictions between pass probability and predicted final grade.
        if prob_pass < 0.35:
            return float(min(raw_grade, 69.0))
        if prob_pass < 0.5:
            return float(min(raw_grade, 74.0))
        if prob_pass < 0.75:
            return float(np.clip(raw_grade, 55.0, 89.0))
        return float(max(raw_grade, 65.0))

    def _prepare_input(self, payload: dict) -> pd.DataFrame:
        row = {k: payload[k] for k in FEATURES}
        return pd.DataFrame([row])

    def predict(self, payload: dict) -> dict:
        x = self._prepare_input(payload)

        prob_pass = float(self.classifier.predict_proba(x)[0, 1])
        baseline_prob = float(self.baseline.predict_proba(x)[0, 1])
        raw_grade = float(np.clip(self.grade_model.predict(x)[0], 0, 100))
        predicted_final_grade = self._coherent_final_grade(prob_pass, raw_grade)
        expectation_gap = predicted_final_grade - float(payload["expected_grade_factor"])

        feature_impacts = self.feature_impacts(x)

        return {
            "probability_of_passing": round(prob_pass, 4),
            "predicted_final_grade": round(predicted_final_grade, 2),
            "expectation_gap": round(expectation_gap, 2),
            "risk_level": self._risk_from_probability(prob_pass),
            "baseline_probability": round(baseline_prob, 4),
            "feature_impacts": feature_impacts,
        }

    def _get_explainer(self):
        if shap is None:
            return None
        if self._explainer is None:
            transformed = self.classifier.named_steps["preprocessor"].transform(self.background_data)
            estimator = self.classifier.named_steps["classifier"]
            self._explainer = shap.Explainer(estimator, transformed)
        return self._explainer

    def feature_impacts(self, x_input: pd.DataFrame) -> dict[str, float]:
        explainer = self._get_explainer()
        if explainer is None:
            return {k: 0.0 for k in FEATURES}

        transformed = self.classifier.named_steps["preprocessor"].transform(x_input)
        shap_values = explainer(transformed)

        values = shap_values.values
        if values.ndim == 3:
            # For binary classifiers where class axis is present.
            values = values[:, :, 1]

        vals = values[0]
        if len(vals) != len(FEATURES):
            # Defensive fallback if transformed feature count differs.
            vals = vals[: len(FEATURES)]

        return {k: round(float(v), 4) for k, v in zip(FEATURES, vals)}


@lru_cache(maxsize=1)
def get_service() -> ModelService:
    return ModelService()
