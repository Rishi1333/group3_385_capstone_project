"""
Docstring for group3_385_capstone_project-main.scripts.fever_recommendation
Train : Fever -> Medicine/Recommendation (classification)
Output artifacts : artifacts/ actifacts_fever_recommendation/
"""
import os 
import json
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

# Config
RANDOM_STATE = 76
TEST_SIZE = 0.2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(
    BASE_DIR,
    "..",
    "datasets",
    "enhanced_fever_medicine_recommendation.csv"
)
ARTIFACTS_DIR = os.path.join(
    BASE_DIR,
    "..",
    "artifacts",
    "artifacts_fever_recommendation"
)
os.makedirs(ARTIFACTS_DIR, exist_ok = True)

# Try to auto-detect label column by common names
LABEL_CANDIDATES = [
    "recommended_medicine", "medicine", "drug", "recommendation", "recommended_medication",
    "medication", "recommended_drug", "label", "target", "output"
]

def pick_label_column(df: pd.DataFrame) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in LABEL_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand]
    # fallback : last column
    return df.columns[-1]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Seperate numeric & categorical columns
    numeric_cols = X.select_dtypes(include = [np.number]).columns.to_list()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numberic_pipe = Pipeline(steps = [
        ("imputer", SimpleImputer(strategy = "median")),
        ("scaler", StandardScaler(with_mean = True))
    ])

    categorical_pipe = Pipeline(steps = [
        ("imputer", SimpleImputer(strategy = "most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown = "ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numberic_pipe,numeric_cols),
            ("cat", categorical_pipe,categorical_cols)
        ],
        remainder = "drop"
    )
    return preprocessor

def evaluate_model(name: str, pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipe.predict(X_test)

    # For multiclass, use macro average
    avg = "macro"
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average = avg, zero_division = 0)),
        "precision_macro": float(precision_score(y_test, y_pred, average = avg, zero_division = 0)),
        "recall_macro": float(recall_score(y_test, y_pred, average = avg, zero_division = 0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division = 0)
    }
    return metrics

def main():
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    
    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Remove columns that are completely empty
    df = df.dropna(axis=1, how="all")

    # Automatically find label column
    label_col = pick_label_column(df)

    # Inject noise into labels (simulate real-world errors)
    NOISE_LEVEL = 0.10   # 10% noise
    n_noise = int(len(df) * NOISE_LEVEL)

    # Randomly select rows for noise injection
    noise_indices = df.sample(n_noise, random_state=RANDOM_STATE).index
    
    # Flip labels only for binary classification
    unique_labels = df[label_col].unique()
    if len(unique_labels) == 2:
        label_map = {
            unique_labels[0]: unique_labels[1],
            unique_labels[1]: unique_labels[0]
        }
        df.loc[noise_indices, label_col] = df.loc[noise_indices, label_col].replace(label_map)

    # Separate features (X) and target (y)
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(str)

    # Remove columns that may cause data leakage
    LEAK_PRONE_COLS = ["Previous_Medication"]
    drop_cols = [c for c in LEAK_PRONE_COLS if c in X.columns]

    if drop_cols:
        X = X.drop(columns=drop_cols)

    # Split dataset into training and testing sets
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(X_train)

    # Define machine learning models
    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "grad_boost": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    all_metrics = []
    best = None

    # Train and evaluate each model
    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", clf)
        ])

        pipe.fit(X_train, y_train)
        metrics = evaluate_model(name, pipe, X_test, y_test)
        all_metrics.append(metrics)

        score = metrics["f1_macro"]
        if best is None or score > best["score"]:
            best = {
                "name": name,
                "score": score,
                "pipe": pipe,
                "metrics": metrics
            }

        joblib.dump(pipe, os.path.join(ARTIFACTS_DIR, f"{name}_pipeline.pkl"))

    # Save experiment configuration
    config = {
        "task": "fever_medicine_recommendation",
        "best_model": {
            "name": best["name"],
            "f1_macro": best["score"]
        },
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "status": "Noise injected for realism"
    }

    with open(os.path.join(ARTIFACTS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Print final results
    print("\n=== Training Finished ===")
    print(f"Best Model: {best['name']} | F1 Score: {best['score']:.4f}")

    # Show leaderboard
    leaderboard = sorted(all_metrics, key=lambda x: x['f1_macro'], reverse=True)
    for m in leaderboard:
        print(f"- {m['model']:15s} Acc: {m['accuracy']:.4f} | F1: {m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()