import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


FEATURES = [
    "study_hours",
    "attendance_pct",
    "assignment_score",
    "midterm_score",
    "sleep_hours",
    "past_gpa",
    "expected_grade_factor",
]


# Synthetic data generator for portfolio/demo use.
def generate_dataset(n_samples: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    study_hours = rng.normal(loc=20, scale=8, size=n_samples).clip(0, 60)
    attendance_pct = rng.normal(loc=82, scale=12, size=n_samples).clip(40, 100)
    assignment_score = rng.normal(loc=76, scale=13, size=n_samples).clip(35, 100)
    midterm_score = rng.normal(loc=73, scale=15, size=n_samples).clip(20, 100)
    sleep_hours = rng.normal(loc=6.9, scale=1.4, size=n_samples).clip(3.5, 10)
    past_gpa = rng.normal(loc=2.8, scale=0.65, size=n_samples).clip(0, 4.0)
    expected_grade_factor = rng.normal(loc=78, scale=10, size=n_samples).clip(40, 100)

    latent = (
        0.06 * study_hours
        + 0.08 * (attendance_pct - 60)
        + 0.10 * (assignment_score - 50)
        + 0.08 * (midterm_score - 50)
        + 1.2 * past_gpa
        + 0.45 * (sleep_hours - 5)
        + 0.03 * (expected_grade_factor - 70)
        - 11.0
    )

    probs = 1 / (1 + np.exp(-latent))
    pass_label = rng.binomial(1, probs)

    # Keep grade targets realistic and correlated with pass probability.
    final_grade = (
        14
        + 0.18 * study_hours
        + 0.18 * attendance_pct
        + 0.26 * assignment_score
        + 0.29 * midterm_score
        + 4.2 * past_gpa
        + 1.1 * sleep_hours
        + 0.08 * expected_grade_factor
        + 18 * (probs - 0.5)
        + rng.normal(0, 6, n_samples)
    ).clip(0, 100)

    df = pd.DataFrame(
        {
            "study_hours": study_hours,
            "attendance_pct": attendance_pct,
            "assignment_score": assignment_score,
            "midterm_score": midterm_score,
            "sleep_hours": sleep_hours,
            "past_gpa": past_gpa,
            "expected_grade_factor": expected_grade_factor,
            "pass": pass_label,
            "final_grade": final_grade,
        }
    )
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, FEATURES)],
        remainder="drop",
    )


def train() -> None:
    df = generate_dataset()
    X = df[FEATURES]
    y = df["pass"]
    y_grade = df["final_grade"]

    X_train, X_test, y_train, y_test, y_grade_train, y_grade_test = train_test_split(
        X, y, y_grade, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()

    logistic = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
            ),
        ]
    )

    logistic.fit(X_train, y_train)
    log_preds = logistic.predict(X_test)
    rf = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    best_model_name = "random_forest"
    best_model = rf
    best_acc = accuracy_score(y_test, rf_preds)

    xgb_report = None
    if HAS_XGBOOST:
        xgb = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=350,
                        max_depth=5,
                        learning_rate=0.07,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_preds)
        xgb_report = {
            "accuracy": xgb_acc,
            "classification_report": classification_report(y_test, xgb_preds, output_dict=True),
        }
        if xgb_acc > best_acc:
            best_model_name = "xgboost"
            best_model = xgb
            best_acc = xgb_acc

    grade_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )
    grade_model.fit(X_train, y_grade_train)
    grade_preds = grade_model.predict(X_test)

    metrics = {
        "baseline_logistic": {
            "accuracy": accuracy_score(y_test, log_preds),
            "classification_report": classification_report(y_test, log_preds, output_dict=True),
        },
        "random_forest": {
            "accuracy": accuracy_score(y_test, rf_preds),
            "classification_report": classification_report(y_test, rf_preds, output_dict=True),
        },
        "xgboost": xgb_report,
        "final_grade_model": {
            "mae": mean_absolute_error(y_grade_test, grade_preds),
        },
        "chosen_classifier": best_model_name,
    }

    # Save artifacts needed by API and SHAP explanations.
    joblib.dump(logistic, ARTIFACT_DIR / "logistic_baseline.joblib")
    joblib.dump(best_model, ARTIFACT_DIR / "classifier.joblib")
    joblib.dump(grade_model, ARTIFACT_DIR / "grade_regressor.joblib")
    joblib.dump(X_train, ARTIFACT_DIR / "background_data.joblib")

    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
