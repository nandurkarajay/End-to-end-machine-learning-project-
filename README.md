# EduPredict — Student Exam Performance Predictor

> An end-to-end Machine Learning web application that predicts a student's **maths score** based on demographic and academic features.

**Built by Ajay Nandurkar**

---

## 🔗 Live Demo

> Deployed on Render → [https://your-render-url.onrender.com](https://your-render-url.onrender.com)

---

## 📌 What This Project Does

A student fills in a simple form with:
- Gender, Race/Ethnicity, Parental Education
- Lunch type, Test Preparation Course
- Reading Score, Writing Score

The app instantly predicts their **Maths Score out of 100** using a trained Machine Learning model.

---

## 🗂️ Project Structure

```
├── app.py                          # Flask web application (entry point)
├── Procfile                        # Gunicorn start command for Render
├── render.yaml                     # Render deployment config
├── requirements.txt                # All Python dependencies
├── setup.py                        # Registers src/ as an installable package
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Reads CSV, splits train/test, saves to artifacts/
│   │   ├── data_transformation.py  # Builds preprocessing pipeline, saves preprocessor.pkl
│   │   └── model_trainer.py        # Trains 7 models, selects best, saves model.pkl
│   │
│   ├── pipeline/
│   │   ├── predict_pipeline.py     # Loads pkl files, runs prediction for web app
│   │   └── train_pipeline.py       # Entry point for training (empty — triggered via data_ingestion.py)
│   │
│   ├── templates/
│   │   ├── index.html              # Landing page
│   │   └── home.html               # Prediction form + result
│   │
│   ├── exception.py                # Custom exception with file name + line number
│   ├── logger.py                   # Timestamped logging to logs/ folder
│   └── utils.py                    # save_object, load_object, evaluate_models
│
├── artifacts/                      # Auto-generated during training (gitignored)
│   ├── model.pkl                   # Best trained model
│   ├── preprocessor.pkl            # Fitted scaler + encoder
│   ├── train.csv
│   └── test.csv
│
└── notebook/
    └── data/
        └── stud.csv                # Raw dataset — 1,000 student records
```

---

## ⚙️ ML Pipeline

### Phase 1 — Training (runs once at deployment)

```
stud.csv
   ↓
data_ingestion.py      → splits 80/20, saves train.csv & test.csv
   ↓
data_transformation.py → scales numerics, encodes categoricals, saves preprocessor.pkl
   ↓
model_trainer.py       → trains 7 models with GridSearchCV, saves best as model.pkl
```

### Phase 2 — Prediction (runs on every user request)

```
User fills form
   ↓
predict_pipeline.py    → loads model.pkl + preprocessor.pkl
   ↓
preprocessor.transform(input)   → same transformation as training
   ↓
model.predict(input)            → returns maths score
   ↓
Result shown on page
```

---

## 🤖 Models Trained & Compared

| Model | Tuned With |
|---|---|
| Random Forest Regressor | `n_estimators` |
| Decision Tree Regressor | `criterion` |
| Gradient Boosting Regressor | `learning_rate`, `subsample`, `n_estimators` |
| Linear Regression | — |
| XGBoost Regressor | `learning_rate`, `n_estimators` |
| CatBoost Regressor | `depth`, `learning_rate`, `iterations` |
| AdaBoost Regressor | `learning_rate`, `n_estimators` |

All models are evaluated using **R² score** on the test set. The best model is automatically selected and saved.

### Best Model Result

| Metric | Value |
|---|---|
| **Model** | Random Forest Regressor |
| **R² Score** | 0.85 |
| **MAE** | 4.65 |
| **n_estimators** | 128 (chosen by GridSearchCV) |

---

## 🧪 Dataset

- **Source:** `notebook/data/stud.csv`
- **Size:** 1,000 rows × 8 columns
- **Target:** `math_score`

| Column | Type | Description |
|---|---|---|
| `gender` | Categorical | male / female |
| `race_ethnicity` | Categorical | group A–E |
| `parental_level_of_education` | Categorical | high school → master's degree |
| `lunch` | Categorical | standard / free/reduced |
| `test_preparation_course` | Categorical | none / completed |
| `reading_score` | Numerical | 0–100 |
| `writing_score` | Numerical | 0–100 |
| `math_score` | Numerical | **Target variable** |

---

## 🔧 Data Preprocessing

**Numerical features** (`reading_score`, `writing_score`):
```
SimpleImputer(strategy='median') → StandardScaler()
```

**Categorical features** (`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`):
```
SimpleImputer(strategy='most_frequent') → OneHotEncoder() → StandardScaler(with_mean=False)
```

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/nandurkarajay/End-to-end-machine-learning-project-.git
cd End-to-end-machine-learning-project-
```

### 2. Create virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python src/components/data_ingestion.py
```
This runs the full training pipeline and saves `model.pkl` + `preprocessor.pkl` to `artifacts/`.

### 5. Start the Flask app
```bash
python app.py
```

Open → [http://localhost:5000](http://localhost:5000)

---

## ☁️ Deployment (Render)

| Setting | Value |
|---|---|
| **Platform** | [Render](https://render.com) |
| **Runtime** | Python 3.11 |
| **Build Command** | `pip install -r requirements.txt && python src/components/data_ingestion.py` |
| **Start Command** | `gunicorn app:application` |
| **Root Directory** | *(leave blank)* |

The build command trains the model on every deploy. The start command serves the Flask app via Gunicorn.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | Flask |
| **ML Library** | Scikit-learn |
| **Boosting Models** | XGBoost, CatBoost |
| **Data Processing** | Pandas, NumPy |
| **Model Persistence** | Pickle |
| **Production Server** | Gunicorn |
| **Frontend** | Bootstrap 5, Bootstrap Icons |
| **Deployment** | Render |
| **Version Control** | Git + GitHub |

---

## 📁 Key Files Explained

| File | Purpose |
|---|---|
| `app.py` | Flask routes — `/` landing page, `/predictdata` prediction form |
| `src/exception.py` | Custom exception that shows exact file + line number of any error |
| `src/logger.py` | Saves timestamped logs to `logs/` folder |
| `src/utils.py` | `save_object`, `load_object`, `evaluate_models` helper functions |
| `src/components/data_ingestion.py` | Run this to trigger the full training pipeline |
| `src/pipeline/predict_pipeline.py` | `PredictPipeline` and `CustomData` classes used by Flask |

---

## 👤 Author

**Ajay Nandurkar**
- GitHub: [@nandurkarajay](https://github.com/nandurkarajay)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
