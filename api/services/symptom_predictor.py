import numpy as np 
import pandas as pd 

class SymptomPredictor():
    
    def __init__(self,model, config = None):
        self.model = model
        if(config):
            self.features = config["features"]
            label_mapping = config["label_mapping"]
            self.label_mappings = {str(k): int(v) for k, v in label_mapping.items()}
            self.inv_label_mapping = {v: k for k, v in self.label_mappings.items()}

    def normalize(self, payload: dict)->dict:
        if "symptoms" in payload and isinstance(payload["symptoms"], dict):
            return payload["symptoms"]
        return payload
    
    def validate(self,symptoms:dict)->pd.DataFrame:
        row = {feat: int(symptoms.get(feat,0)) for feat in self.features}
        return pd.DataFrame([row])
    
    def predict(self,payload:dict)-> dict:
        symptoms = self.normalize(payload)
        X = self.validate(symptoms)

        probs = self.model.predict_proba(X)[0]
        top_idx = int(np.argmax(probs))

        prob_list = sorted(
            [(self.inv_label_mapping[i], float(p)) for i, p in enumerate(probs)],
            key = lambda x:x[1], 
            reverse = True
        )

        return{
            "predictions": self.inv_label_mapping[top_idx],
            "probabilities":prob_list
        }
