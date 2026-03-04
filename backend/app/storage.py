from __future__ import annotations

import csv
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from .model_service import ARTIFACT_DIR

HISTORY_FIELDS = [
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
]


class PredictionStorage(ABC):
    @abstractmethod
    def save_prediction(self, payload: dict, result: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_recent(self, limit: int = 100) -> list[dict]:
        raise NotImplementedError


class CSVPredictionStorage(PredictionStorage):
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def save_prediction(self, payload: dict, result: dict) -> None:
        write_header = not self.file_path.exists()
        with open(self.file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(_history_row(payload, result))

    def list_recent(self, limit: int = 100) -> list[dict]:
        if not self.file_path.exists():
            return []
        with open(self.file_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows.reverse()
        return rows[:limit]


class SQLitePredictionStorage(PredictionStorage):
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    course_name TEXT NOT NULL,
                    study_hours REAL NOT NULL,
                    attendance_pct REAL NOT NULL,
                    assignment_score REAL NOT NULL,
                    midterm_score REAL NOT NULL,
                    sleep_hours REAL NOT NULL,
                    past_gpa REAL NOT NULL,
                    expected_grade_factor REAL NOT NULL,
                    probability_of_passing REAL NOT NULL,
                    predicted_final_grade REAL NOT NULL,
                    expectation_gap REAL NOT NULL,
                    risk_level TEXT NOT NULL
                )
                """
            )

    def save_prediction(self, payload: dict, result: dict) -> None:
        row = _history_row(payload, result)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_history (
                    timestamp_utc,
                    course_name,
                    study_hours,
                    attendance_pct,
                    assignment_score,
                    midterm_score,
                    sleep_hours,
                    past_gpa,
                    expected_grade_factor,
                    probability_of_passing,
                    predicted_final_grade,
                    expectation_gap,
                    risk_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["timestamp_utc"],
                    row["course_name"],
                    row["study_hours"],
                    row["attendance_pct"],
                    row["assignment_score"],
                    row["midterm_score"],
                    row["sleep_hours"],
                    row["past_gpa"],
                    row["expected_grade_factor"],
                    row["probability_of_passing"],
                    row["predicted_final_grade"],
                    row["expectation_gap"],
                    row["risk_level"],
                ),
            )

    def list_recent(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    timestamp_utc,
                    course_name,
                    study_hours,
                    attendance_pct,
                    assignment_score,
                    midterm_score,
                    sleep_hours,
                    past_gpa,
                    expected_grade_factor,
                    probability_of_passing,
                    predicted_final_grade,
                    expectation_gap,
                    risk_level
                FROM prediction_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


def _history_row(payload: dict, result: dict) -> dict:
    return {
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


def build_storage() -> PredictionStorage:
    backend = (
        __import__("os").environ.get("PREDICTION_STORAGE_BACKEND", "sqlite").strip().lower()
    )
    if backend == "csv":
        return CSVPredictionStorage(ARTIFACT_DIR / "prediction_history.csv")
    return SQLitePredictionStorage(ARTIFACT_DIR / "prediction_history.db")
