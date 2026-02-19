import numpy as np 
import pandas as pd 

class HeartPredictor:
    """
    Predictor class for heart disease prediction model.
    Handles feature validation, prediction, and probability output.
    """
    
    def __init__(self, model, config=None):
        self.model = model
        if config:
            self.features = config["features"]
            self.numeric_features = config.get("numeric_features", [])
            self.categorical_features = config.get("categorical_features", [])
            self.categorical_encodings = config.get("categorical_encodings", {})
            label_mapping = config.get("label_mapping", {})
            self.label_mappings = {str(k): int(v) for k, v in label_mapping.items()}
            self.inv_label_mapping = {v: k for k, v in self.label_mappings.items()}

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
        Handles both numeric and categorical features.
        """
        row = {}
        for feat in self.features:
            val = features.get(feat, 0)
            
            # Handle categorical features - convert string values to numeric
            if feat in self.categorical_encodings:
                mapping = self.categorical_encodings[feat].get("mapping", {})
                if isinstance(val, str):
                    val = mapping.get(val.lower(), mapping.get(val, 0))
                row[feat] = int(val)
            else:
                # Numeric feature
                try:
                    row[feat] = float(val)
                except (ValueError, TypeError):
                    row[feat] = 0.0
        
        return pd.DataFrame([row])
    
    def predict(self, payload: dict) -> dict:
        """
        Run prediction and return results with probabilities.
        """
        features = self.normalize(payload)
        X = self.validate(features)

        # Get prediction probabilities
        probs = self.model.predict_proba(X)[0]
        top_idx = int(np.argmax(probs))

        # Build probability list sorted by probability
        prob_list = sorted(
            [(self.inv_label_mapping.get(i, f"class_{i}"), float(p)) 
             for i, p in enumerate(probs)],
            key=lambda x: x[1], 
            reverse=True
        )

        return {
            "predictions": self.inv_label_mapping.get(top_idx, f"class_{top_idx}"),
            "probabilities": prob_list,
            "model_type": "heart_disease"
        }
