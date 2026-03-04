# AI Student Performance Predictor (EdTech)

A portfolio-ready ML web app for early student academic risk detection.

## What it does
Students enter:
- Course name
- Study hours
- Attendance %
- Assignment scores
- Midterm score
- Sleep hours
- Past GPA
- Expected grade factor

The app predicts:
- Probability of passing
- Predicted final grade
- Risk level (Low/Medium/High)

Includes:
- Logistic Regression baseline
- Random Forest and XGBoost candidate models (best selected by test accuracy)
- SHAP feature impact output for explainability
- FastAPI backend + React frontend

## Project structure

- `backend/ml/train.py`: model training pipeline + artifact generation
- `backend/app/main.py`: API endpoints (`/predict`, `/metrics`, `/health`)
- `backend/app/model_service.py`: inference and SHAP impact generation
- `frontend/src/App.jsx`: input form and prediction dashboard
- `backend/artifacts/prediction_history.csv`: saved course-level prediction history

## Quick start

### 1) Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ml.train
uvicorn app.main:app --reload --port 8000
```

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "course_name": "Data Structures",
    "study_hours": 22,
    "attendance_pct": 88,
    "assignment_score": 81,
    "midterm_score": 76,
    "sleep_hours": 7.2,
    "past_gpa": 3.1,
    "expected_grade_factor": 84
  }'
```

## Deployment outline

### Backend (Render)
- Create a Web Service.
- Root directory: `backend`.
- Build command: `pip install -r requirements.txt && python -m ml.train`.
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

### Frontend (Render static or Vercel)
- Root directory: `frontend`.
- Build command: `npm install && npm run build`.
- Publish directory: `dist`.
- Set API base URL to backend URL (proxy/env-based).

### AWS option
- Backend: ECS/Fargate or EC2 with Docker.
- Frontend: S3 + CloudFront.

## Portfolio pitch

"Built a deployed ML system predicting student academic risk with explainable AI (SHAP), combining a logistic regression baseline with tree-based models. Delivered pass-probability, final-grade forecasting, and actionable risk segmentation for advisors and tutors."

Replace with your measured metric gain:
- `Improved prediction accuracy by X% over baseline models.`

## Notes
- Current training uses synthetic data for demo/portfolio scaffolding.
- Replace with real institutional LMS data for production use.
