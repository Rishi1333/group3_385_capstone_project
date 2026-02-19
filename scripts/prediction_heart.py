# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:13:55 2026

@author: rishi
"""

import os 
import json
import numpy as np
import pandas as pd 
import joblib
#import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score,f1_score, precision_score,recall_score,confusion_matrix,classification_report

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



# local dataset copy extracted into `datasets`
PATH  = r"your_path_to_datasets/heart_disease_uci.csv"
# prefer dataset in repository datasets folder; if not present, fall back to PATH or try to extract from zips
PROJECT_DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
LOCAL_HEART_CSV = os.path.join(PROJECT_DATASETS_DIR, "heart_disease_uci.csv")
if os.path.exists(LOCAL_HEART_CSV):
    DATASET_PATH = LOCAL_HEART_CSV
else:
    DATASET_PATH = PATH

# save outputs to the repository artifacts folder (heart-specific subfolder)
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "artifacts_heart")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
OUTPUT_DIR = ARTIFACTS_DIR

RANDOM_STATE = 27

numeric_features = ["age", "trestbps", "chol","thalch", "oldpeak", "ca"]
categorical_features = ["sex","dataset","cp", "fbs", "restecg", "exang", "slope", "thal"]
TARGET_COLUMN = "num"
heavy_missing_columns = ["ca","slope","thal"]

def to_numeric(df,cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def evaluate_binary_data(y_true,y_proba,y_pred):
    return {
        "roc":float(roc_auc_score(y_true,y_proba)),
        "f1":float(f1_score(y_true,y_pred)),
        "precision":float(precision_score(y_true,y_pred)),
        "recall":float(recall_score(y_true,y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }

# load dataset: prefer extracted local CSV; otherwise try to read from zips in the datasets folder
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded heart dataset from {DATASET_PATH}")
else:
    # try to locate and read the CSV from zip archives in the datasets folder
    found = False
    if os.path.isdir(PROJECT_DATASETS_DIR):
        for fname in os.listdir(PROJECT_DATASETS_DIR):
            if fname.lower().endswith('.zip'):
                zpath = os.path.join(PROJECT_DATASETS_DIR, fname)
                try:
                    import zipfile
                    with zipfile.ZipFile(zpath) as z:
                        for info in z.infolist():
                            if info.filename.lower().endswith('heart_disease_uci.csv') or os.path.basename(info.filename).lower()=='heart_disease_uci.csv':
                                with z.open(info) as f:
                                    df = pd.read_csv(f)
                                    print(f"Loaded heart dataset from {zpath}:{info.filename}")
                                    found = True
                                    break
                except Exception:
                    continue
            if found:
                break
    if not found:
        raise FileNotFoundError(f"Could not find heart dataset at {DATASET_PATH} or inside project zips")

missing = [c for c in (numeric_features+categorical_features+[TARGET_COLUMN]) if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns in the dataset: {missing}")
    
df = to_numeric(df, numeric_features)
df = df.dropna(subset= [TARGET_COLUMN])

y = (df[TARGET_COLUMN].astype(float)>0).astype(int)

X = df[numeric_features+categorical_features].copy()


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE, stratify=y)


#=====================
#Preprocessing PipeLine
#=====================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='median', add_indicator=True)),
    ("scaler", StandardScaler())
    ])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

preprocesser = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer , numeric_features),
        ("cat", categorical_transformer, categorical_features)
        ],remainder = "drop")


#=====================
# Model 
#=====================

models = {
    
    "logreg":LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
        ),
    "random_forest":RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample"
        ),
    "grad_boost":GradientBoostingClassifier(random_state=RANDOM_STATE)
    
}

results = {}

for name,model in models.items():
    clf = Pipeline(steps=[
        ("preprocesser", preprocesser),
        ("model",model)
        ])
    
    clf.fit(X_train,y_train)
    
    
    try:
        y_proba= clf.predict_proba(X_test)[:,1]
    except Exception:
        
        try:
            y_proba= clf.decision_function(X_test)
        except Exception:
            y_proba= clf.predict(X_test)
    
    
    y_pred = clf.predict(X_test)
    metrics = evaluate_binary_data(y_test,y_proba, y_pred)
    results[name] = metrics
    
    joblib.dump(clf,os.path.join(OUTPUT_DIR,f"{name}_pipeline.pkl"))
    
config = {
    "data_path": PATH,
    "target": "binary(num>0)",
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "test_size": 0.2,
}

# record environment versions to make future unpickling deterministic
import sklearn, sys, platform
config["sklearn_version"] = sklearn.__version__
config["python_version"] = sys.version
config["platform"] = platform.platform()

with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open (os.path.join(OUTPUT_DIR,"metrics.json"), "w") as f:
    json.dump(results,f,indent=2)
    

# Print summary
print("\n==================== RESULTS (Test Set) ====================")
for name, m in results.items():
    print(f"\nModel: {name}")
    print(f"  ROC-AUC   : {m['roc']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  Confusion : {m['confusion_matrix']}")

print(f"\nSaved outputs to: {OUTPUT_DIR}")
print("Artifacts saved:")
print(" - logreg_pipeline.pkl")
print(" - random_forest_pipeline.pkl")
print(" - grad_boost_pipeline.pkl")
print(" - metrics.json")
print(" - config.json")








# Load trained Random Forest pipeline
rf_pipeline = joblib.load(
    r"your_path_to_artifacts/artifacts_heart/random_forest_pipeline.pkl"
)

# Get feature names after preprocessing
preprocessor = rf_pipeline.named_steps["preprocesser"]
model = rf_pipeline.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
importances = model.feature_importances_

# Create importance dataframe
fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Top 15 features
print(fi_df.head(15))

# Plot
#





# Load Logistic Regression pipeline
logreg_pipeline = joblib.load(
    r"your_path_to_artifacts/artifacts_heart/logreg_pipeline.pkl"
)

preprocessor = logreg_pipeline.named_steps["preprocesser"]
model = logreg_pipeline.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
coefficients = model.coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values(by="coefficient", ascending=False)

print("Top risk-increasing features:")
print(coef_df.head(10))

print("\nTop protective features:")
print(coef_df.tail(10))























    
