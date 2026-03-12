"""
Mental Health Predictor Service
Handles mental health screening using trained sklearn pipelines.
"""
import numpy as np
import pandas as pd


class MentalHealthPredictor:
    """
    Predictor class for mental health screening model.
    Handles feature validation, prediction, and probability output.
    
    Features expected (21 total):
    - Age: int
    - Gender: str (male/female/other)
    - family_history: str (Yes/No)
    - work_interfere: str (Often/Rarely/Never/Sometimes/etc)
    - no_employees: str (1-5/6-25/26-100/100-500/500-1000/More than 1000)
    - remote_work: str (Yes/No)
    - tech_company: str (Yes/No)
    - benefits: str (Yes/No/Don't know)
    - care_options: str (Yes/No/Not sure)
    - wellness_program: str (Yes/No/Don't know)
    - seek_help: str (Yes/No/Don't know)
    - anonymity: str (Yes/No/Don't know)
    - leave: str (Very easy/Somewhat easy/Don't know/Somewhat difficult/Very difficult)
    - mental_health_consequence: str (Yes/No/Maybe)
    - phys_health_consequence: str (Yes/No/Maybe)
    - coworkers: str (Yes/No/Some of them)
    - supervisor: str (Yes/No/Some of them)
    - mental_health_interview: str (Yes/No/Maybe)
    - phys_health_interview: str (Yes/No/Maybe)
    - mental_vs_physical: str (Yes/No/Don't know)
    - obs_consequence: str (Yes/No)
    """
    
    # Default feature list
    DEFAULT_FEATURES = [
        "Age", "Gender", "family_history", "work_interfere", "no_employees",
        "remote_work", "tech_company", "benefits", "care_options", "wellness_program",
        "seek_help", "anonymity", "leave", "mental_health_consequence",
        "phys_health_consequence", "coworkers", "supervisor", "mental_health_interview",
        "phys_health_interview", "mental_vs_physical", "obs_consequence"
    ]
    
    def __init__(self, model, config=None):
        self.model = model
        if config:
            self.features = config.get("features", self.DEFAULT_FEATURES)
            self.numeric_features = config.get("numeric_features", ["Age"])
            self.categorical_features = config.get("categorical_features", 
                [f for f in self.features if f not in ["Age"]])
        else:
            self.features = self.DEFAULT_FEATURES
            self.numeric_features = ["Age"]
            self.categorical_features = [f for f in self.features if f not in ["Age"]]
    
    def normalize(self, payload: dict) -> dict:
        """
        Normalize payload to extract features dict.
        Supports both {'features': {...}} and direct feature dict.
        """
        if "features" in payload and isinstance(payload["features"], dict):
            return payload["features"]
        return payload
    
    def validate(self, features: dict) -> pd.DataFrame:
        """
        Validate and prepare features for model input.
        The trained pipeline includes a ColumnTransformer that handles:
        - Numeric features: imputation + scaling
        - Categorical features: imputation + one-hot encoding
        """
        row = {}
        for feat in self.features:
            val = features.get(feat, None)
            
            if feat in self.numeric_features:
                # Numeric feature
                try:
                    row[feat] = float(val) if val is not None else np.nan
                except (ValueError, TypeError):
                    row[feat] = np.nan
            else:
                # Categorical feature - keep as string
                row[feat] = str(val) if val is not None else "Unknown"
        
        return pd.DataFrame([row])
    
    def predict(self, payload: dict) -> dict:
        """
        Run prediction and return results with probabilities.
        
        Returns:
            dict with:
            - prediction: "treatment_recommended" or "no_treatment"
            - probability: float (probability of needing treatment)
            - model_type: "mental_health"
        """
        features = self.normalize(payload)
        X = self.validate(features)
        
        # Get prediction probabilities
        probs = self.model.predict_proba(X)[0]
        
        # Binary classification: [no_treatment, treatment]
        no_treatment_prob = float(probs[0])
        treatment_prob = float(probs[1])
        
        prediction = "treatment_recommended" if treatment_prob >= 0.5 else "no_treatment"
        confidence = max(treatment_prob, no_treatment_prob)
        
        return {
            "prediction": prediction,
            "probability": treatment_prob,
            "confidence": confidence,
            "probabilities": [
                ("no_treatment", no_treatment_prob),
                ("treatment_recommended", treatment_prob)
            ],
            "model_type": "mental_health",
            "features_used": list(features.keys())
        }
    
    def get_risk_factors(self, features: dict) -> list:
        """
        Identify potential risk factors based on input features.
        Returns a list of concerning indicators.
        """
        risk_factors = []
        
        # Family history
        if features.get("family_history", "").lower() == "yes":
            risk_factors.append("Family history of mental health issues")
        
        # Work interference
        work_interfere = features.get("work_interfere", "").lower()
        if work_interfere in ["often", "sometimes"]:
            risk_factors.append("Mental health interferes with work")
        
        # Lack of employer support
        if features.get("benefits", "").lower() in ["no", "don't know"]:
            risk_factors.append("No mental health benefits at work")
        
        if features.get("care_options", "").lower() in ["no", "not sure"]:
            risk_factors.append("Limited awareness of care options")
        
        # Observed consequences
        if features.get("obs_consequence", "").lower() == "yes":
            risk_factors.append("Observed negative consequences for mental health at workplace")
        
        return risk_factors