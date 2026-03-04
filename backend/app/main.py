import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model_service import ARTIFACT_DIR, get_service
from .schemas import PredictionOutput, StudentInput

app = FastAPI(title="AI Student Performance Predictor", version="1.0.0")
HISTORY_FILE = ARTIFACT_DIR / "prediction_history.csv"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics() -> dict:
    metrics_path = ARTIFACT_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Run training first.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: StudentInput) -> PredictionOutput:
    required = [
        ARTIFACT_DIR / "classifier.joblib",
        ARTIFACT_DIR / "logistic_baseline.joblib",
        ARTIFACT_DIR / "grade_regressor.joblib",
        ARTIFACT_DIR / "background_data.joblib",
    ]
    if not all(path.exists() for path in required):
        raise HTTPException(status_code=404, detail="Model artifacts missing. Run `python -m ml.train` first.")

    service = get_service()
    payload = input_data.model_dump()
    result = service.predict(payload)

    write_header = not HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "course_name",
                "study_hours",
                "attendance_pct",
                "assignment_score",
                "midterm_score",
                "sleep_hours",
                "past_gpa",
                "expected_grade_factor",
                "probability_of_passing",
                "predicted_final_grade",
                "expectation_gap",
                "risk_level",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "course_name": payload["course_name"],
                "study_hours": payload["study_hours"],
                "attendance_pct": payload["attendance_pct"],
                "assignment_score": payload["assignment_score"],
                "midterm_score": payload["midterm_score"],
                "sleep_hours": payload["sleep_hours"],
                "past_gpa": payload["past_gpa"],
                "expected_grade_factor": payload["expected_grade_factor"],
                "probability_of_passing": result["probability_of_passing"],
                "predicted_final_grade": result["predicted_final_grade"],
                "expectation_gap": result["expectation_gap"],
                "risk_level": result["risk_level"],
            }
        )

    return PredictionOutput(**result)
