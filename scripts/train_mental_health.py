# ============================================================
# Mental Health Screening - Training Pipeline
# File: train_mental_health.py
# ============================================================
"""
Training script for mental health screening from survey data.
Uses mixed categorical + numerical features.

Dataset: datasets/mental_health/survey.csv
Output: artifacts/artifacts_mental_health/
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Try to import XGBoost and CatBoost (optional, falls back if not installed)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Note: XGBoost not installed. Using alternative models.")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Note: CatBoost not installed. Using alternative models.")


# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "datasets", "mental_health", "survey.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts", "artifacts_mental_health")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Target column
TARGET_COLUMN = "treatment"

# Features to use (excluding free-text and identifier columns)
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "family_history",
    "work_interfere",
    "no_employees",
    "remote_work",
    "tech_company",
    "benefits",
    "care_options",
    "wellness_program",
    "seek_help",
    "anonymity",
    "leave",
    "mental_health_consequence",
    "phys_health_consequence",
    "coworkers",
    "supervisor",
    "mental_health_interview",
    "phys_health_interview",
    "mental_vs_physical",
    "obs_consequence"
]

# Columns to exclude (identifiers, free text, etc.)
EXCLUDE_COLUMNS = ["Timestamp", "state", "self_employed", "Country", "comments"]


# =========================
# LOAD DATA
# =========================
print("Loading dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Remove excluded columns
df = df.drop(columns=[c for c in EXCLUDE_COLUMNS if c in df.columns], errors="ignore")

# Update feature columns to only include available columns
available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
print(f"\nUsing {len(available_features)} features: {available_features}")


# =========================
# PREPROCESS
# =========================
print("\nPreprocessing...")

# Clean Gender column (normalize variations)
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].str.lower().str.strip()
    gender_map = {
        "male": "male",
        "m": "male",
        "male-ish": "male",
        "maile": "male",
        "mal": "male",
        "male (cis)": "male",
        "cis male": "male",
        "man": "male",
        "female": "female",
        "f": "female",
        "female (cis)": "female",
        "cis female": "female",
        "woman": "female",
        "femake": "female",
    }
    df["Gender"] = df["Gender"].map(lambda x: gender_map.get(x, "other"))

# Clean Age column (remove outliers)
if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df.loc[df["Age"] < 18, "Age"] = np.nan
    df.loc[df["Age"] > 100, "Age"] = np.nan
    median_age = df["Age"].median()
    df["Age"] = df["Age"].fillna(median_age)

# Encode target
if TARGET_COLUMN in df.columns:
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
else:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")

print(f"\nTarget distribution:")
print(y.value_counts())
print(f"Treatment seeking rate: {y.mean():.2%}")

# Extract features
X = df[available_features].copy()
print(f"Features shape: {X.shape}")

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")


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
# PREPROCESSING PIPELINE
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop"
)


# =========================
# MODELS
# =========================
# Using diverse models better suited for mixed categorical + numerical survey data
models = {
    "mlp": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
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
        max_depth=15,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    ),
}

# Add XGBoost if available (excellent for mixed feature types)
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

# Add CatBoost if available (excellent for categorical features)
if HAS_CATBOOST:
    models["catboost"] = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        random_state=RANDOM_STATE,
        verbose=False
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
    
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
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
    "task": "mental_health_screening",
    "dataset": {
        "path": DATASET_PATH,
        "n_samples": len(df),
        "n_features": len(available_features),
        "target_distribution": {
            "no_treatment": int((y == 0).sum()),
            "treatment": int((y == 1).sum())
        }
    },
    "best_model": {
        "name": best_model_name,
        "roc_auc" if "roc_auc" in results[best_model_name] else "f1": float(best_score)
    },
    "features": available_features,
    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,
    "target": TARGET_COLUMN,
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
print(f"  Features: {len(available_features)} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
print(f"  Treatment seeking: {(y == 1).sum()} ({y.mean():.1%})")

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
