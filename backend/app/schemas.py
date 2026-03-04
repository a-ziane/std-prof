from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    course_name: str = Field(..., min_length=1, max_length=120)
    study_hours: float = Field(..., ge=0, le=80)
    attendance_pct: float = Field(..., ge=0, le=100)
    assignment_score: float = Field(..., ge=0, le=100)
    midterm_score: float = Field(..., ge=0, le=100)
    sleep_hours: float = Field(..., ge=0, le=16)
    past_gpa: float = Field(..., ge=0, le=4)
    expected_grade_factor: float = Field(..., ge=0, le=100)


class PredictionOutput(BaseModel):
    probability_of_passing: float
    predicted_final_grade: float
    expectation_gap: float
    risk_level: str
    baseline_probability: float
    feature_impacts: dict[str, float]


class PredictionHistoryItem(BaseModel):
    timestamp_utc: str
    course_name: str
    study_hours: float
    attendance_pct: float
    assignment_score: float
    midterm_score: float
    sleep_hours: float
    past_gpa: float
    expected_grade_factor: float
    probability_of_passing: float
    predicted_final_grade: float
    expectation_gap: float
    risk_level: str
