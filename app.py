import os
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__, template_folder=os.path.join('src', 'templates'))
app = application

# Path where prediction history is stored
HISTORY_FILE = os.path.join("artifacts", "prediction_history.csv")

# CSV columns
HISTORY_COLUMNS = [
    "timestamp",
    "gender",
    "ethnicity",
    "parental_education",
    "lunch",
    "test_prep",
    "reading_score",
    "writing_score",
    "predicted_score"
]


def save_prediction(input_data: dict, predicted_score: float):
    """Append one prediction row to the history CSV."""
    row = {
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gender":            input_data["gender"],
        "ethnicity":         input_data["ethnicity"],
        "parental_education": input_data["parental_level_of_education"],
        "lunch":             input_data["lunch"],
        "test_prep":         input_data["test_preparation_course"],
        "reading_score":     input_data["reading_score"],
        "writing_score":     input_data["writing_score"],
        "predicted_score":   round(float(predicted_score), 2)
    }
    df_row = pd.DataFrame([row])

    if os.path.exists(HISTORY_FILE):
        df_row.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        df_row.to_csv(HISTORY_FILE, mode='w', header=True, index=False)


def load_history(n: int = 10):
    """Load the last n predictions from the history CSV."""
    if not os.path.exists(HISTORY_FILE):
        return []
    df = pd.read_csv(HISTORY_FILE)
    # Return last n rows, newest first
    recent = df.tail(n).iloc[::-1]
    return recent.to_dict(orient='records')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    history = load_history(10)

    if request.method == 'GET':
        return render_template('home.html', history=history)

    # ── POST: run prediction ──────────────────────────────────────────────────
    form = {
        "gender":                    request.form.get('gender'),
        "ethnicity":                 request.form.get('ethnicity'),
        "parental_level_of_education": request.form.get('parental_level_of_education'),
        "lunch":                     request.form.get('lunch'),
        "test_preparation_course":   request.form.get('test_preparation_course'),
        "reading_score":             float(request.form.get('reading_score')),
        "writing_score":             float(request.form.get('writing_score'))
    }

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

    # Save to history
    save_prediction(form, score)

    # Reload history so the new row appears immediately
    history = load_history(10)

    return render_template('home.html', results=score, history=history)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
