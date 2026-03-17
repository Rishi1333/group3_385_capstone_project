"""
Diabetes Predictor Service
Handles diabetes prediction using trained sklearn pipelines.
"""
import numpy as np
import pandas as pd


class DiabetesPredictor:
    """
    Predictor class for diabetes prediction model.
    Handles feature validation, prediction, and probability output.
    
    Features expected:
    - Pregnancies: int
    - Glucose: int
    - BloodPressure: int
    - SkinThickness: int
    - Insulin: int
    - BMI: float
    - DiabetesPedigreeFunction: float
    - Age: int
    """
    
    def __init__(self, model, config=None):
        self.model = model
        if config:
            self.features = config.get("features", [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ])
            self.zero_as_missing = config.get("preprocessing", {}).get(
                "zero_as_missing_columns", 
                ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            )
        else:
            self.features = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]
            self.zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
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
        Handles missing values and type conversion.
        """
        row = {}
        for feat in self.features:
            val = features.get(feat, 0)
            
            # Convert to float
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
            
            # Handle zero as missing for certain columns
            # Note: The trained pipeline handles imputation internally
            row[feat] = val
        
        return pd.DataFrame([row])
    
    def predict(self, payload: dict) -> dict:
        """
        Run prediction and return results with probabilities.
        
        Returns:
            dict with:
            - prediction: "diabetic" or "non_diabetic"
            - probability: float (probability of diabetic)
            - model_type: "diabetes"
        """
        features = self.normalize(payload)
        X = self.validate(features)
        
        # Get prediction probabilities
        probs = self.model.predict_proba(X)[0]
        
        # Assuming binary classification: [non_diabetic, diabetic]
        non_diabetic_prob = float(probs[0])
        diabetic_prob = float(probs[1])
        
        prediction = "diabetic" if diabetic_prob >= 0.5 else "non_diabetic"
        confidence = max(diabetic_prob, non_diabetic_prob)
        
        return {
            "prediction": prediction,
            "probability": diabetic_prob,
            "confidence": confidence,
            "probabilities": [
                ("non_diabetic", non_diabetic_prob),
                ("diabetic", diabetic_prob)
            ],
            "model_type": "diabetes",
            "features_used": list(features.keys())
        }
    
    def get_feature_importance(self) -> dict:
        """Get feature importance if available from the model."""
        if hasattr(self.model, 'named_steps'):
            # Pipeline model
            for name, step in self.model.named_steps.items():
                if hasattr(step, 'feature_importances_'):
                    return dict(zip(self.features, step.feature_importances_.tolist()))
                elif hasattr(step, 'coef_'):
                    return dict(zip(self.features, step.coef_[0].tolist()))
        return {}
