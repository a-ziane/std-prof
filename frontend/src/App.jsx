import { useMemo, useState } from 'react';

const defaultForm = {
  course_name: '',
  study_hours: 20,
  attendance_pct: 85,
  assignment_score: 78,
  midterm_score: 75,
  sleep_hours: 7,
  past_gpa: 3.0,
  expected_grade_factor: 82,
};

function Field({ label, name, min, max, step = 0.1, value, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        name={name}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
        required
      />
    </label>
  );
}

function ImpactChart({ impacts }) {
  const rows = Object.entries(impacts || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  if (!rows.length) return null;

  const max = Math.max(...rows.map(([, v]) => Math.abs(v))) || 1;

  return (
    <div className="impacts">
      <h3>SHAP Feature Impacts</h3>
      {rows.map(([feature, value]) => {
        const width = `${(Math.abs(value) / max) * 100}%`;
        const positive = value >= 0;
        return (
          <div key={feature} className="impact-row">
            <div className="impact-head">
              <span>{feature}</span>
              <span>{value.toFixed(3)}</span>
            </div>
            <div className="impact-bar-wrap">
              <div className={`impact-bar ${positive ? 'positive' : 'negative'}`} style={{ width }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const riskClass = useMemo(() => {
    if (!result) return '';
    return result.risk_level.toLowerCase();
  }, [result]);

  const onChange = (e) => {
    const { name, value } = e.target;
    if (name === 'course_name') {
      setForm((prev) => ({ ...prev, [name]: value }));
      return;
    }
    setForm((prev) => ({ ...prev, [name]: Number(value) }));
  };

  const runPrediction = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const [predRes, metricsRes] = await Promise.all([
        fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(form),
        }),
        fetch('/api/metrics'),
      ]);

      if (!predRes.ok) {
        const data = await predRes.json();
        throw new Error(data.detail || 'Prediction failed');
      }

      const predData = await predRes.json();
      setResult(predData);

      if (metricsRes.ok) {
        const metricsData = await metricsRes.json();
        setMetrics(metricsData);
      }
    } catch (err) {
      setError(err.message || 'Unexpected error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="layout">
      <section className="card">
        <h1>AI Student Performance Predictor</h1>
        <p>Estimate pass probability, predicted final grade, and risk using explainable ML.</p>

        <form className="form" onSubmit={runPrediction}>
          <label className="field">
            <span>Course Name</span>
            <input
              type="text"
              name="course_name"
              maxLength={120}
              value={form.course_name}
              onChange={onChange}
              placeholder="e.g. Data Structures"
              required
            />
          </label>
          <Field label="Study Hours / Week" name="study_hours" min={0} max={80} value={form.study_hours} onChange={onChange} />
          <Field label="Attendance %" name="attendance_pct" min={0} max={100} value={form.attendance_pct} onChange={onChange} />
          <Field label="Assignment Score" name="assignment_score" min={0} max={100} value={form.assignment_score} onChange={onChange} />
          <Field label="Midterm Score" name="midterm_score" min={0} max={100} value={form.midterm_score} onChange={onChange} />
          <Field label="Sleep Hours / Night" name="sleep_hours" min={0} max={16} value={form.sleep_hours} onChange={onChange} />
          <Field label="Past GPA" name="past_gpa" min={0} max={4} step={0.01} value={form.past_gpa} onChange={onChange} />
          <Field
            label="Expected Grade Factor"
            name="expected_grade_factor"
            min={0}
            max={100}
            value={form.expected_grade_factor}
            onChange={onChange}
          />

          <button type="submit" disabled={loading}>{loading ? 'Predicting...' : 'Predict Performance'}</button>
        </form>
      </section>

      <section className="card">
        <h2>Prediction Output</h2>
        {error && <p className="error">{error}</p>}

        {!result && !error && <p>Enter values and run prediction.</p>}

        {result && (
          <div className="results">
            <div className="metric-grid">
              <article>
                <span>Probability of Passing</span>
                <strong>{(result.probability_of_passing * 100).toFixed(1)}%</strong>
              </article>
              <article>
                <span>Predicted Final Grade</span>
                <strong>{result.predicted_final_grade.toFixed(1)}</strong>
              </article>
              <article className={`risk ${riskClass}`}>
                <span>Risk Level</span>
                <strong>{result.risk_level}</strong>
              </article>
              <article>
                <span>Logistic Baseline</span>
                <strong>{(result.baseline_probability * 100).toFixed(1)}%</strong>
              </article>
              <article>
                <span>Expectation Gap</span>
                <strong>{result.expectation_gap.toFixed(1)}</strong>
              </article>
            </div>
            <ImpactChart impacts={result.feature_impacts} />
          </div>
        )}

        {metrics && (
          <div className="metrics">
            <h3>Model Benchmark</h3>
            <p>Chosen model: <strong>{metrics.chosen_classifier}</strong></p>
            <p>Baseline accuracy: <strong>{(metrics.baseline_logistic.accuracy * 100).toFixed(2)}%</strong></p>
            <p>Random Forest accuracy: <strong>{(metrics.random_forest.accuracy * 100).toFixed(2)}%</strong></p>
            {metrics.xgboost && (
              <p>XGBoost accuracy: <strong>{(metrics.xgboost.accuracy * 100).toFixed(2)}%</strong></p>
            )}
          </div>
        )}
      </section>
    </main>
  );
}
