# ============================================================
# Diabetes Prediction - Training Pipeline
# File: train_diabetes.py
# ============================================================
"""
Training script for diabetes risk prediction from clinical features.
Uses numerical features with standard scaling.

Dataset: datasets/diabetes/diabetes.csv
Output: artifacts/artifacts_diabetes/
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Try to import XGBoost and LightGBM (optional, falls back if not installed)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Note: XGBoost not installed. Using alternative models.")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Note: LightGBM not installed. Using alternative models.")


# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "datasets", "diabetes", "diabetes.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts", "artifacts_diabetes")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Feature definitions
NUMERIC_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]
TARGET_COLUMN = "Outcome"


# =========================
# LOAD DATA
# =========================
print("Loading dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for missing values
print(f"\nMissing values per column:")
print(df.isnull().sum())

# Basic statistics
print(f"\nTarget distribution:")
print(df[TARGET_COLUMN].value_counts())
print(f"Diabetes positive rate: {df[TARGET_COLUMN].mean():.2%}")


# =========================
# PREPROCESS
# =========================
print("\nPreprocessing...")

# Handle zeros as missing values for certain features (common in medical data)
# Glucose, BloodPressure, SkinThickness, Insulin, BMI should not be 0
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_as_missing:
    if col in df.columns:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            print(f"  {col}: {zero_count} zeros (treated as missing)")
            df[col] = df[col].replace(0, np.nan)

# Fill missing values with median
for col in NUMERIC_FEATURES:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# Extract features and target
X = df[NUMERIC_FEATURES].copy()
y = df[TARGET_COLUMN].astype(int)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")


# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")


# =========================
# MODELS
# =========================
# Using diverse models better suited for small tabular medical data
models = {
    "mlp": MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False
    ),
    "svm": SVC(
        C=1.0,
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    ),
}

# Add XGBoost if available (excellent for tabular data)
if HAS_XGBOOST:
    models["xgboost"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        use_label_encoder=False
    )

# Add LightGBM if available (fast and efficient for tabular data)
if HAS_LIGHTGBM:
    models["lightgbm"] = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )

results = {}


# =========================
# TRAIN + EVALUATE
# =========================
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline with scaling
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Probabilities for ROC-AUC
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc_auc = None
    
    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    if roc_auc is not None:
        metrics["roc_auc"] = float(roc_auc)
    
    results[name] = metrics
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:   {metrics['recall']:.4f}")
    if roc_auc:
        print(f"  ROC-AUC:  {roc_auc:.4f}")
    
    # Save pipeline
    pipeline_path = os.path.join(ARTIFACTS_DIR, f"{name}_pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)
    print(f"  Saved: {pipeline_path}")


# =========================
# FIND BEST MODEL
# =========================
best_model_name = max(results, key=lambda x: results[x].get("roc_auc", results[x]["f1"]))
best_score = results[best_model_name].get("roc_auc", results[best_model_name]["f1"])
print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name} (Score: {best_score:.4f})")
print(f"{'='*60}")


# =========================
# SAVE CONFIG
# =========================
config = {
    "task": "diabetes_prediction",
    "dataset": {
        "path": DATASET_PATH,
        "n_samples": len(df),
        "n_features": len(NUMERIC_FEATURES),
        "target_distribution": {
            "negative": int((df[TARGET_COLUMN] == 0).sum()),
            "positive": int((df[TARGET_COLUMN] == 1).sum())
        }
    },
    "best_model": {
        "name": best_model_name,
        "roc_auc" if "roc_auc" in results[best_model_name] else "f1": float(best_score)
    },
    "features": NUMERIC_FEATURES,
    "target": TARGET_COLUMN,
    "preprocessing": {
        "zero_as_missing_columns": zero_as_missing,
        "missing_value_strategy": "median"
    },
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE
}

# Record environment versions
import sklearn
import sys
import platform
config["sklearn_version"] = sklearn.__version__
config["python_version"] = sys.version
config["platform"] = platform.platform()


# Convert numpy types for JSON serialization
def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "item"):
        return obj.item()
    else:
        return obj


config_safe = to_python(config)
results_safe = to_python(results)

# Save config
config_path = os.path.join(ARTIFACTS_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config_safe, f, indent=2)
print(f"\nSaved config: {config_path}")

# Save metrics
metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results_safe, f, indent=2)
print(f"Saved metrics: {metrics_path}")


# =========================
# PRINT SUMMARY
# =========================
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"\nDataset Statistics:")
print(f"  Total samples: {len(df)}")
print(f"  Features: {len(NUMERIC_FEATURES)}")
print(f"  Diabetes positive: {(df[TARGET_COLUMN] == 1).sum()} ({df[TARGET_COLUMN].mean():.1%})")

print(f"\nModel Performance (Test Set):")
for name, m in sorted(results.items(), key=lambda x: x[1].get("roc_auc", x[1]["f1"]), reverse=True):
    roc_str = f" | ROC-AUC: {m['roc_auc']:.4f}" if "roc_auc" in m else ""
    print(f"  {name:15s} - Acc: {m['accuracy']:.4f} | F1: {m['f1']:.4f}{roc_str}")

print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")
print("Files created:")
for fname in os.listdir(ARTIFACTS_DIR):
    fpath = os.path.join(ARTIFACTS_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  - {fname} ({size_kb:.1f} KB)")
