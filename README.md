=======
# Comp 385 capstone project

## Disease & Prediction Prototype

This repository contains training, inference, and an API for disease ML pipelines:

- Disease prediction from symptoms 

This README documents project layout, how to build the environment, run training, run the API, where artifacts are stored, and common troubleshooting steps for collaborators.

---

**Repository root**

- `prediction_disease_prediction.py` — training script for the symptoms/disease models (logistic regression, random forest). Saves artifacts under `artifacts/artifacts_disease_prediction/`.
- `inference.ipynb` — notebook with quick inference and model-loading examples.
- `requirements.txt` — pinned project requirements (team-chosen scikit-learn version kept as-is).

**Top-level folders**

- `api/` — lightweight Flask API serving the models.
  - `app.py` — application factory and blueprint registration
  - `config/` — static config files (e.g. `symptom_mapping.json`)
  - `routes/` — blueprint route handlers (predict endpoints)
  - `services/` — helpers: artifact loader, predictors, feature extractor
- `artifacts/` — model artifacts and metadata are saved here after training. Example subfolder:
  - `artifacts_disease_prediction/` — contains `*_pipeline.pkl`, `config.json`, `metrics.json` for disease prediction
- `datasets/` — CSVs used for training

---

## Getting started

Create a virtual environment (PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Training

- Disease models:
  - Script: `prediction_disease_prediction.py`
  - Input: `datasets/disease_prediction.csv`
  - Output: artifacts saved to `artifacts/artifacts_disease_prediction/` (pipelines, `config.json`, `metrics.json`).
  - Use: `python prediction_disease_prediction.py`

---

## Notes when training

- Each training script writes a `config.json` and `metrics.json`. We have added environment metadata to `config.json` (sklearn version, python version, platform) to help avoid unpickle problems.
- If you change `requirements.txt` or the scikit-learn version, retrain models in that environment.

---

## API (Flask)

- Location: `api/app.py` (factory) and `api/routes/*` (blueprints).
- How it loads models: `api/services/artifact_loader.py` reads pre-trained pickles from `artifacts/`.

Start API (with venv active):

```powershell
python api\app.py
```

The app in the repository is configured to run on `0.0.0.0:3000` by default.

---

## Endpoints (examples)

- POST /predict/symptoms
  - Content-Type: `application/json`
  - Payload: either a flat mapping of symptom binary features or a wrapper: `{ "symptoms": { ... } }`.
  - Successful response: predicted disease label and probability list.

---

## Testing the API locally

1. Activate venv and start the API from project root:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python api\app.py
   ```
2. Use `curl` or Postman to call the endpoints above.

---

## Recommended git workflow for models

Models should be stored under `artifacts/`

 
