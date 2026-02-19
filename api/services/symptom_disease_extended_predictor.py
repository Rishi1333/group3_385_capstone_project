"""
Symptom-Disease Extended Predictor Service
Handles disease prediction from symptom text using NLP-based sklearn pipelines.
"""
import numpy as np
import pandas as pd
import joblib
import os


class SymptomDiseaseExtendedPredictor:
    """
    Predictor class for symptom-based disease prediction using NLP.
    Uses TF-IDF vectorization and multi-class classification.
    
    Input: Symptom text (e.g., "I have headache, fever, and body aches")
    Output: Predicted disease with probabilities
    """
    
    def __init__(self, model, config=None, label_encoder=None):
        self.model = model
        self.label_encoder = label_encoder
        
        if config:
            self.label_mapping = config.get("label_mapping", {})
            # Create inverse mapping (index -> disease name)
            self.inv_label_mapping = {v: k for k, v in self.label_mapping.items()}
        else:
            self.label_mapping = {}
            self.inv_label_mapping = {}
    
    @classmethod
    def load(cls, artifacts_dir: str):
        """
        Load model, config, and label encoder from artifacts directory.
        """
        # Load config
        config_path = os.path.join(artifacts_dir, "config.json")
        config = None
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
        
        # Load label encoder
        label_encoder_path = os.path.join(artifacts_dir, "label_encoder.pkl")
        label_encoder = None
        if os.path.exists(label_encoder_path):
            label_encoder = joblib.load(label_encoder_path)
        
        # Load best model (try MLP first, then others)
        model = None
        for model_name in ["mlp_pipeline.pkl", "random_forest_pipeline.pkl", 
                          "linear_svc_pipeline.pkl", "logreg_pipeline.pkl"]:
            model_path = os.path.join(artifacts_dir, model_name)
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                break
        
        if model is None:
            raise FileNotFoundError(f"No model found in {artifacts_dir}")
        
        return cls(model=model, config=config, label_encoder=label_encoder)
    
    def normalize(self, payload: dict) -> str:
        """
        Normalize payload to extract symptom text.
        Supports various input formats.
        """
        # Direct text input
        if isinstance(payload, str):
            return payload
        
        # Dict with 'text' key
        if "text" in payload:
            return payload["text"]
        
        # Dict with 'symptoms' key
        if "symptoms" in payload:
            if isinstance(payload["symptoms"], str):
                return payload["symptoms"]
            elif isinstance(payload["symptoms"], list):
                return " ".join(payload["symptoms"])
        
        # Dict with 'features' key
        if "features" in payload:
            if isinstance(payload["features"], str):
                return payload["features"]
            elif isinstance(payload["features"], dict):
                # Convert feature dict to text
                parts = []
                for k, v in payload["features"].items():
                    if v and str(v).lower() not in ["no", "none", "false", "0"]:
                        parts.append(k.replace("_", " "))
                return " ".join(parts)
        
        # Fallback: convert entire dict to string
        return str(payload)
    
    def predict(self, payload: dict) -> dict:
        """
        Run prediction and return results with probabilities.
        
        Returns:
            dict with:
            - prediction: str (predicted disease name)
            - confidence: float
            - probabilities: list of (disease, probability) tuples
            - model_type: "symptom_disease_extended"
        """
        text = self.normalize(payload)
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba([text])[0]
            
            # Get top prediction
            top_idx = int(np.argmax(probs))
            
            # Build probability list (top 5)
            prob_items = [(self.inv_label_mapping.get(i, f"disease_{i}"), float(p)) 
                         for i, p in enumerate(probs)]
            prob_items.sort(key=lambda x: x[1], reverse=True)
            top_probabilities = prob_items[:5]
            
            prediction = self.inv_label_mapping.get(top_idx, f"disease_{top_idx}")
            confidence = float(probs[top_idx])
        else:
            # For models without predict_proba (e.g., LinearSVC)
            pred = self.model.predict([text])[0]
            prediction = self.inv_label_mapping.get(pred, f"disease_{pred}")
            confidence = 1.0
            top_probabilities = [(prediction, 1.0)]
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": top_probabilities,
            "model_type": "symptom_disease_extended",
            "input_text": text[:200] + "..." if len(text) > 200 else text
        }
    
    def predict_from_symptoms_list(self, symptoms: list) -> dict:
        """
        Convenience method to predict from a list of symptoms.
        
        Args:
            symptoms: List of symptom strings (e.g., ["headache", "fever", "nausea"])
        
        Returns:
            Prediction result dict
        """
        text = " ".join(symptoms)
        return self.predict({"symptoms": text})
    
    def get_similar_diseases(self, text: str, top_k: int = 10) -> list:
        """
        Get top-k similar diseases for a given symptom text.
        
        Args:
            text: Symptom description
            top_k: Number of results to return
        
        Returns:
            List of (disease_name, probability) tuples
        """
        if not hasattr(self.model, 'predict_proba'):
            pred = self.model.predict([text])[0]
            return [(self.inv_label_mapping.get(pred, f"disease_{pred}"), 1.0)]
        
        probs = self.model.predict_proba([text])[0]
        prob_items = [(self.inv_label_mapping.get(i, f"disease_{i}"), float(p)) 
                     for i, p in enumerate(probs)]
        prob_items.sort(key=lambda x: x[1], reverse=True)
        
        return prob_items[:top_k]