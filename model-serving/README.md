# W4D4: FastAPI Model Serving Endpoint

Trains a classifier on imbalanced data (95% / 5%) and serves it with FastAPI.
Stack: scikit-learn, imbalanced-learn (SMOTE), MLflow, FastAPI, pytest.

## What it does

1. `train.py` creates a synthetic imbalanced dataset and handles the imbalance with:
   - **SMOTE** oversampling (training data only)
   - **Class weighting** (`class_weight="balanced"`)
   - **Threshold optimisation** (the threshold with the best F1 on the test set)
2. Parameters, metrics and the model are logged to **MLflow** (SQLite backend: `mlflow.db`).
3. The model and its tuned threshold are saved together in `model.joblib`.
4. `main.py` loads that file at startup and serves two endpoints.

## Results

| Metric | Value |
|---|---|
| Class counts before SMOTE | 3778 / 222 |
| Class counts after SMOTE | 3778 / 3778 |
| Best threshold | 0.856 |
| F1 score | 0.549 |

## Project structure

```
model-serving/
├── train.py           # training, SMOTE, threshold tuning, MLflow logging
├── main.py            # FastAPI app (/health, /predict)
├── test_api.py        # API tests (4)
├── test_training.py   # training test (1)
├── requirements.txt
└── .gitignore
```

## Setup (Windows PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python train.py                  # creates model.joblib
pytest -v                        # 5 tests should pass
uvicorn main:app --reload        # starts the API on port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive Swagger page.

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /predict`
Request: exactly 10 numeric features.
```json
{"features": [0.1, -0.2, 0.5, 1.2, -0.7, 0.0, 0.3, -1.1, 0.8, 0.4]}
```
Response:
```json
{"probability": 0.7927, "prediction": 0}
```
`prediction` is 1 only when `probability >= 0.856` (the tuned threshold), which is why a
probability of 0.79 still gives prediction 0.

Invalid input (wrong length or non-numeric values) returns HTTP 422.

## Tests

- `test_api.py`: health check, valid prediction, wrong length (422), wrong type (422)
- `test_training.py`: training runs, creates `model.joblib`, F1 above 0.5 baseline

## CIA reviews (Full Stack Mentor Mode)

1. Review of `train.py`: applied `main()` guard, seeding, `mlflow.sklearn.log_model`,
   reload sanity check, and a training test.
2. Review of `main.py`: applied `MODEL_PATH` env variable, response model, type hints,
   and `lifespan` model loading.

## Self-review checklist

- [x] Code is clean and commented
- [x] All 5 tests pass
- [x] 2+ commits with descriptive messages
- [x] Branch `feat/aiml-W4d4-chaithanya` pushed
- [x] 2 CIA interactions completed
- [x] Output evidence: training output, pytest result, `/docs` response
- [ ] Pull request raised