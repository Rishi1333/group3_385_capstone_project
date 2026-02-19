# ============================================================
# Disease Prediction from Symptoms - Training Pipeline
# File: train_symptom_disease.py
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier


# =========================
# CONFIG
# =========================
DATA_PATH = r"your_path_to_datasets/disease_prediction.csv"
ARTIFACTS_DIR = r"your_path_to_artifacts/artifacts_disease_prediction"
OUTPUT_DIR = ARTIFACTS_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
# Assume last column is target
TARGET_COL = df.columns[-1]
FEATURE_COLS = df.columns[:-1]

X = df[FEATURE_COLS].astype(int)
y_raw = df[TARGET_COL].astype(str)

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Save label mapping
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# =========================
# MODELS
# =========================
models = {
    "logreg_multiclass": LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs"
    ),
    "bernoulli_nb": BernoulliNB(),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE
    )
}

results = {}

# =========================
# TRAIN + EVALUATE
# =========================
for name, model in models.items():
    clf = Pipeline(steps=[
        ("model", model)
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )
    }

    results[name] = metrics

    # Save pipeline
    joblib.dump(clf, os.path.join(OUTPUT_DIR, f"{name}_pipeline.pkl"))

# Save config + metrics
config = {
    "data_path": DATA_PATH,
    "target": TARGET_COL,
    "features": list(FEATURE_COLS),
    "label_mapping": label_map,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE
}

# record environment versions to make future unpickling deterministic
import sklearn, sys, platform
config["sklearn_version"] = sklearn.__version__
config["python_version"] = sys.version
config["platform"] = platform.platform()

def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()   # converts numpy scalars
    else:
        return obj

config_safe = to_python(config)

with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config_safe, f, indent=2)


with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n==================== RESULTS (Test Set) ====================")
for name, m in results.items():
    print(f"\nModel: {name}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  Macro F1 : {m['macro_f1']:.4f}")

print(f"\nSaved outputs to: {OUTPUT_DIR}")
print("Artifacts saved:")
for name in models:
    print(f" - {name}_pipeline.pkl")
print(" - metrics.json")
print(" - config.json")
