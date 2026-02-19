# ============================================================
# Symptom-Disease Extended - NLP-based Disease Prediction
# File: train_symptom_disease_extended.py
# ============================================================
"""
Training script for NLP-based disease prediction from natural language
symptom descriptions. Uses TF-IDF vectorization + classification.

Dataset: datasets/Symptom-Disease Extended/
Output: artifacts/artifacts_symptom_disease_extended/
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# Try to import XGBoost (optional, falls back if not installed)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Note: XGBoost not installed. Using alternative models.")


# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "datasets", "Symptom-Disease Extended")
TRAIN_PATH = os.path.join(DATASET_DIR, "symptom-disease-train-dataset.csv")
TEST_PATH = os.path.join(DATASET_DIR, "symptom-disease-test-dataset.csv")
MAPPING_PATH = os.path.join(DATASET_DIR, "mapping.json")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts", "artifacts_symptom_disease_extended")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.15  # Additional validation split from training data


# =========================
# LOAD DATA
# =========================
print("Loading datasets...")

# Load training data
train_df = pd.read_csv(TRAIN_PATH)
print(f"Training samples: {len(train_df)}")

# Load test data
test_df = pd.read_csv(TEST_PATH)
print(f"Test samples: {len(test_df)}")

# Load label mapping (disease ID -> disease name)
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    label_mapping = json.load(f)
print(f"Disease labels in mapping: {len(label_mapping)}")

# Extract text and labels
X_train_text = train_df["text"].astype(str).values
y_train_raw = train_df["label"].astype(str).values

X_test_text = test_df["text"].astype(str).values
y_test_raw = test_df["label"].astype(str).values

# Encode labels
label_encoder = LabelEncoder()
# Fit on all unique labels from both train and test
all_labels = np.concatenate([y_train_raw, y_test_raw])
label_encoder.fit(all_labels)

y_train = label_encoder.transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

# Create reverse mapping (encoded label -> disease name)
encoded_to_disease = {}
for i, class_label in enumerate(label_encoder.classes_):
    disease_name = label_mapping.get(class_label, f"Unknown_Disease_{class_label}")
    encoded_to_disease[str(i)] = disease_name

print(f"Total unique diseases: {len(label_encoder.classes_)}")


# =========================
# SPLIT TRAINING DATA FOR VALIDATION
# =========================
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_text, y_train,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)
print(f"Training split: {len(X_train_split)}, Validation: {len(X_val)}")


# =========================
# MODELS
# =========================
# Using diverse models better suited for NLP text classification
models = {
    "mlp": MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False
    ),
    "linear_svc": LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=RANDOM_STATE
    ),
    "multinomial_nb": MultinomialNB(alpha=0.1),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    ),
}

# Add XGBoost if available (excellent for text classification with TF-IDF)
if HAS_XGBOOST:
    models["xgboost"] = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        use_label_encoder=False
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
    
    # Create pipeline with TF-IDF
    pipeline = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            max_features=10000,  # Limit vocabulary size
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True  # Apply sublinear TF scaling
        )),
        ("model", model)
    ])
    
    # Train
    pipeline.fit(X_train_split, y_train_split)
    
    # Evaluate on validation set
    y_val_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="weighted")
    
    # Evaluate on test set
    y_test_pred = pipeline.predict(X_test_text)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    
    metrics = {
        "validation": {
            "accuracy": float(val_accuracy),
            "weighted_f1": float(val_f1)
        },
        "test": {
            "accuracy": float(test_accuracy),
            "weighted_f1": float(test_f1),
            "macro_f1": float(f1_score(y_test, y_test_pred, average="macro")),
            "precision_weighted": float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0))
        }
    }
    
    results[name] = metrics
    
    print(f"  Validation - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
    print(f"  Test       - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
    
    # Save pipeline
    pipeline_path = os.path.join(ARTIFACTS_DIR, f"{name}_pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)
    print(f"  Saved: {pipeline_path}")


# =========================
# FIND BEST MODEL
# =========================
best_model_name = max(results, key=lambda x: results[x]["test"]["weighted_f1"])
best_f1 = results[best_model_name]["test"]["weighted_f1"]
print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name} (Test F1: {best_f1:.4f})")
print(f"{'='*60}")


# =========================
# SAVE CONFIG
# =========================
config = {
    "task": "symptom_disease_prediction_nlp",
    "dataset": {
        "train_path": TRAIN_PATH,
        "test_path": TEST_PATH,
        "mapping_path": MAPPING_PATH,
        "n_train_samples": len(train_df),
        "n_test_samples": len(test_df),
        "n_unique_diseases": len(label_encoder.classes_)
    },
    "best_model": {
        "name": best_model_name,
        "test_weighted_f1": float(best_f1),
        "test_accuracy": float(results[best_model_name]["test"]["accuracy"])
    },
    "tfidf_params": {
        "max_features": 10000,
        "ngram_range": [1, 2],
        "stop_words": "english",
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True
    },
    "label_mapping": label_mapping,
    "encoded_to_disease": encoded_to_disease,
    "label_encoder_classes": label_encoder.classes_.tolist(),
    "random_state": RANDOM_STATE,
    "validation_split": TEST_SIZE
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
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config_safe, f, indent=2, ensure_ascii=False)
print(f"\nSaved config: {config_path}")

# Save metrics
metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(results_safe, f, indent=2)
print(f"Saved metrics: {metrics_path}")

# Save label encoder
encoder_path = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
joblib.dump(label_encoder, encoder_path)
print(f"Saved label encoder: {encoder_path}")


# =========================
# PRINT SUMMARY
# =========================
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"\nDataset Statistics:")
print(f"  Training samples: {len(train_df)}")
print(f"  Test samples: {len(test_df)}")
print(f"  Unique diseases: {len(label_encoder.classes_)}")

print(f"\nModel Performance (Test Set):")
for name, m in sorted(results.items(), key=lambda x: x[1]["test"]["weighted_f1"], reverse=True):
    print(f"  {name:15s} - Acc: {m['test']['accuracy']:.4f} | F1: {m['test']['weighted_f1']:.4f}")

print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")
print("Files created:")
for fname in os.listdir(ARTIFACTS_DIR):
    fpath = os.path.join(ARTIFACTS_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  - {fname} ({size_kb:.1f} KB)")
