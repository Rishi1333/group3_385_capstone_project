import json
import joblib
import os

class ArtifactLoader:
    """
    Loads ML model artifacts and configurations from disk.
    Supports multiple model types with their respective configurations.
    """

    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir

    def load_symptom_bundle(self) -> dict:
        """
        Load the symptom prediction model bundle.
        Returns dict with 'model' and 'config' keys.
        """
        model_path = os.path.join(
            self.artifacts_dir,
            "artifacts_disease_prediction",
            "logreg_multiclass_pipeline.pkl"
        )
        config_path = os.path.join(
            self.artifacts_dir,
            "artifacts_disease_prediction",
            "config.json"
        )
        
        model = joblib.load(model_path)
        with open(config_path, "r") as f:
            config = json.load(f)
            
        return {"model": model, "config": config}

    def load_heart_bundle(self) -> dict:
        """
        Load the heart disease prediction model bundle.
        Returns dict with 'model' and 'config' keys.
        """
        model_path = os.path.join(
            self.artifacts_dir,
            "artifcats_heart_prediction",  # Note: keeping original folder name with typo
            "heart_random_forest_pipeline.pkl"
        )
        config_path = os.path.join(
            self.artifacts_dir,
            "artifcats_heart_prediction",
            "config.json"
        )
        
        model = joblib.load(model_path)
        with open(config_path, "r") as f:
            config = json.load(f)
            
        return {"model": model, "config": config}

    def load_fever_bundle(self) -> dict:
        """
        Load the fever recommendation model bundle.
        Returns dict with 'model' and 'config' keys.
        """
        model_path = os.path.join(
            self.artifacts_dir,
            "artifacts_fever_recommendation",
            "random_forest_pipeline.pkl"
        )
        config_path = os.path.join(
            self.artifacts_dir,
            "artifacts_fever_recommendation",
            "config.json"
        )
        
        model = joblib.load(model_path)
        with open(config_path, "r") as f:
            config = json.load(f)
            
        return {"model": model, "config": config}

    def load_diabetes_bundle(self) -> dict:
        """
        Load the diabetes prediction model bundle.
        Returns dict with 'model' and 'config' keys.
        """
        model_path = os.path.join(
            self.artifacts_dir,
            "artifacts_diabetes",
            "random_forest_pipeline.pkl"
        )
        config_path = os.path.join(
            self.artifacts_dir,
            "artifacts_diabetes",
            "config.json"
        )
        
        model = joblib.load(model_path)
        with open(config_path, "r") as f:
            config = json.load(f)
            
        return {"model": model, "config": config}

    def load_mental_health_bundle(self) -> dict:
        """
        Load the mental health screening model bundle.
        Returns dict with 'model' and 'config' keys.
        """
        model_path = os.path.join(
            self.artifacts_dir,
            "artifacts_mental_health",
            "adaboost_pipeline.pkl"
        )
        config_path = os.path.join(
            self.artifacts_dir,
            "artifacts_mental_health",
            "config.json"
        )
        
        # The mental health model is stored as a dict with 'model', 'label_encoders', etc.
        bundle = joblib.load(model_path)
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Extract the actual model from the bundle
        if isinstance(bundle, dict) and "model" in bundle:
            model = bundle["model"]
            # Store additional components in config
            config["label_encoders"] = bundle.get("label_encoders", {})
            config["age_scaler"] = bundle.get("age_scaler")
            config["feature_cols"] = bundle.get("feature_cols", [])
        else:
            model = bundle
            
        return {"model": model, "config": config}

    def load_all_bundles(self) -> dict:
        """
        Load all available model bundles.
        Returns dict with bundle names as keys.
        """
        bundles = {}
        
        try:
            bundles["symptom"] = self.load_symptom_bundle()
        except FileNotFoundError:
            pass
            
        try:
            bundles["heart"] = self.load_heart_bundle()
        except FileNotFoundError:
            pass
            
        try:
            bundles["fever"] = self.load_fever_bundle()
        except FileNotFoundError:
            pass
        
        try:
            bundles["diabetes"] = self.load_diabetes_bundle()
        except FileNotFoundError:
            pass
        
        try:
            bundles["mental_health"] = self.load_mental_health_bundle()
        except FileNotFoundError:
            pass
            
        return bundles
