import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score, roc_auc_score
)

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "datasets", "heart_disease_uci.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts", "artifcats_heart_prediction")
OUTPUT_DIR = ARTIFACTS_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv(DATA_PATH)

# ----------------------------
# 2. Handle missing values
# ----------------------------
df.replace("?", np.nan, inplace=True)
df = df.dropna()

# ----------------------------
# 3. Define features and target
# ----------------------------
# Numeric features
numeric_features = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
# Categorical features (excluding id which is not a feature)
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
TARGET_COLUMN = "num"

# Convert numeric columns
for col in numeric_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ----------------------------
# 4. Convert target to binary
# ----------------------------
df[TARGET_COLUMN] = (df[TARGET_COLUMN].astype(float) > 0).astype(int)

# ----------------------------
# 5. Encode categorical features
# ----------------------------
label_encoders = {}
for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = {
            "classes": le.classes_.tolist(),
            "mapping": dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))
        }

# ----------------------------
# 6. Split features and label
# ----------------------------
FEATURE_COLS = numeric_features + categorical_features
X = df[FEATURE_COLS]
y = df[TARGET_COLUMN]

# ----------------------------
# 7. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ----------------------------
# 8. Create pipeline
# ----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))
])

# ----------------------------
# 9. Train model
# ----------------------------
pipeline.fit(X_train, y_train)

# ----------------------------
# 10. Evaluate
# ----------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
}

if y_proba is not None:
    metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)

results = {"random_forest": metrics}

# ----------------------------
# 11. Save config
# ----------------------------
config = {
    "data_path": DATA_PATH,
    "target": TARGET_COLUMN,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "features": FEATURE_COLS,
    "label_mapping": {"no_disease": 0, "disease": 1},
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "categorical_encodings": label_encoders

}

# Record environment versions
import sklearn, sys, platform
config["sklearn_version"] = sklearn.__version__
config["python_version"] = sys.version
config["platform"] = platform.platform()

# Convert numpy types to Python types for JSON serialization
def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    else:
        return obj

config_safe = to_python(config)
metrics_safe = to_python(results)

with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config_safe, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_safe, f, indent=2)

# ----------------------------
# 12. Save model pipeline
# ----------------------------
joblib.dump(pipeline, os.path.join(OUTPUT_DIR, "heart_random_forest_pipeline.pkl"))

# Print summary
print("\n==================== RESULTS (Test Set) ====================")
print(f"\nModel: random_forest")
print(f"  Accuracy  : {metrics['accuracy']:.4f}")
print(f"  F1        : {metrics['f1']:.4f}")
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
if 'roc_auc' in metrics:
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")

print(f"\nSaved outputs to: {OUTPUT_DIR}")
print("Artifacts saved:")
print(" - heart_random_forest_pipeline.pkl")
print(" - metrics.json")
print(" - config.json")
