import os
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__, template_folder=os.path.join('src', 'templates'))
app = application

# ── Constants ─────────────────────────────────────────────────────────────────
HISTORY_FILE  = os.path.join("artifacts", "prediction_history.csv")
DATASET_FILE  = os.path.join("notebook", "data", "stud.csv")

HISTORY_COLUMNS = [
    "timestamp", "gender", "ethnicity", "parental_education",
    "lunch", "test_prep", "reading_score", "writing_score", "predicted_score"
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_prediction(form: dict, predicted_score: float):
    """Append one prediction row to the history CSV."""
    row = {
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gender":             form["gender"],
        "ethnicity":          form["ethnicity"],
        "parental_education": form["parental_level_of_education"],
        "lunch":              form["lunch"],
        "test_prep":          form["test_preparation_course"],
        "reading_score":      form["reading_score"],
        "writing_score":      form["writing_score"],
        "predicted_score":    round(float(predicted_score), 2)
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(HISTORY_FILE):
        df_row.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        df_row.to_csv(HISTORY_FILE, mode='w', header=True, index=False)


def load_history(n: int = 10) -> list:
    """Return the last n predictions, newest first."""
    if not os.path.exists(HISTORY_FILE):
        return []
    df = pd.read_csv(HISTORY_FILE)
    return df.tail(n).iloc[::-1].to_dict(orient='records')


def load_dataset(n: int = 100) -> tuple:
    """Return first n rows of the dataset + summary stats."""
    df = pd.read_csv(DATASET_FILE)
    stats = {
        "total":       len(df),
        "avg_math":    round(df["math_score"].mean(), 1),
        "avg_reading": round(df["reading_score"].mean(), 1),
        "avg_writing": round(df["writing_score"].mean(), 1),
    }
    rows = df.head(n).to_dict(orient='records')
    return rows, stats


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    history = load_history(10)

    if request.method == 'GET':
        return render_template('home.html', history=history)

    # Read form
    form = {
        "gender":                      request.form.get('gender'),
        "ethnicity":                   request.form.get('ethnicity'),
        "parental_level_of_education": request.form.get('parental_level_of_education'),
        "lunch":                       request.form.get('lunch'),
        "test_preparation_course":     request.form.get('test_preparation_course'),
        "reading_score":               float(request.form.get('reading_score')),
        "writing_score":               float(request.form.get('writing_score'))
    }

    # Predict
    data = CustomData(
        gender=form["gender"],
        race_ethnicity=form["ethnicity"],
        parental_level_of_education=form["parental_level_of_education"],
        lunch=form["lunch"],
        test_preparation_course=form["test_preparation_course"],
        reading_score=form["reading_score"],
        writing_score=form["writing_score"]
    )
    pred_df  = data.get_data_as_data_frame()
    pipeline = PredictPipeline()
    results  = pipeline.predict(pred_df)
    score    = results[0]

    # Persist
    save_prediction(form, score)
    history = load_history(10)

    return render_template('home.html', results=score, history=history)


@app.route('/dataset')
def dataset():
    rows, stats = load_dataset(100)
    return render_template('dataset.html', dataset=rows, stats=stats)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
