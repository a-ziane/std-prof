import json
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model_service import ARTIFACT_DIR, get_service
from .schemas import PredictionHistoryItem, PredictionOutput, StudentInput
from .storage import PredictionStorage, build_storage

app = FastAPI(title="AI Student Performance Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_storage() -> PredictionStorage:
    return build_storage()


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


@app.get("/history", response_model=list[PredictionHistoryItem])
def get_history(limit: int = 50) -> list[PredictionHistoryItem]:
    safe_limit = max(1, min(limit, 500))
    rows = get_storage().list_recent(limit=safe_limit)
    return [PredictionHistoryItem(**row) for row in rows]


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
    get_storage().save_prediction(payload, result)

    return PredictionOutput(**result)
