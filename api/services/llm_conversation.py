import ollama
import re
import json
from typing import List, Dict, Any, Optional


class LLMConversation:
    """
    Handles LLM-based conversation for medical triage.
    Generates questions and explanations for different model types.
    """

    # Regex pattern to clean thinking tags from LLM output
    THINKING_PATTERN = r"<think>.*?</think>"

    def _clean(self, text: str) -> str:
        """Clean LLM output by removing thinking tags and extra whitespace."""
        text = re.sub(self.THINKING_PATTERN, "", text, flags=re.S).strip()
        return text

    def bulk_questions(self, features_to_ask: List[str], known_positive: List[str], model_type: str = "symptom") -> List[Dict]:
        """
        Generate yes/no questions for each feature to ask.
        
        Args:
            features_to_ask: List of features to generate questions for
            known_positive: List of already detected positive features
            model_type: Type of model ("symptom" or "heart")
            
        Returns:
            List of dicts: [{"feature": "cough", "question": "Are you experiencing a cough?"}, ...]
        """
        if model_type == "heart":
            return self._heart_questions(features_to_ask, known_positive)
        else:
            return self._symptom_questions(features_to_ask, known_positive)

    def _symptom_questions(self, features_to_ask: List[str], known_positive: List[str]) -> List[Dict]:
        """Generate questions for symptom model features."""
        prompt = f"""
            You are a medical triage assistant.

            KNOWN PRESENT SYMPTOMS: {known_positive}

            TASK:
            Generate ONE yes/no question for EACH symptom in FEATURES_TO_ASK.

            FEATURES_TO_ASK (do not add new symptoms):
            {features_to_ask}

            RULES:
            - Return ONLY valid JSON (no markdown).
            - Output must be a JSON array of objects.
            - Each object must have keys: "feature" and "question".
            - Each question must be ONE sentence and end with '?'.
            - Each question must mention ONLY that feature (no extra symptoms).
            - Use second-person wording (example: "Are you experiencing abdominal pain?").
            - Every question must be natural and unique.

            Return ONLY the JSON array.
            """

        resp = ollama.chat(
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": prompt}]
        )

        raw = self._clean(resp["message"]["content"])

        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\[\s*{.*}\s*\]", raw, flags=re.S)
            if not m:
                return [{"feature": f, "question": f"Do you have {f.replace('_', ' ')}?"} for f in features_to_ask]
            try:
                return json.loads(m.group(0))
            except Exception:
                return [{"feature": f, "question": f"Do you have {f.replace('_', ' ')}?"} for f in features_to_ask]

    def _heart_questions(self, features_to_ask: List[str], known_positive: List[str]) -> List[Dict]:
        """Generate questions for heart model features."""
        # Feature descriptions for better question generation
        feature_descriptions = {
            "age": "your age in years",
            "sex": "your biological sex",
            "cp": "chest pain type (typical angina, atypical angina, non-anginal pain, or asymptomatic)",
            "trestbps": "your resting blood pressure in mm Hg",
            "chol": "your serum cholesterol level in mg/dl",
            "fbs": "whether your fasting blood sugar is greater than 120 mg/dl",
            "restecg": "your resting electrocardiographic results",
            "thalch": "your maximum heart rate achieved",
            "exang": "whether you experience exercise-induced angina",
            "oldpeak": "ST depression induced by exercise relative to rest",
            "slope": "the slope of your peak exercise ST segment",
            "ca": "number of major vessels colored by fluoroscopy",
            "thal": "thalassemia status (normal, fixed defect, or reversable defect)"
        }

        prompt = f"""
            You are a medical triage assistant specializing in cardiac health assessment.

            KNOWN INFORMATION: {known_positive if known_positive else 'None'}

            TASK:
            Generate ONE question for EACH feature in FEATURES_TO_ASK to collect the needed medical information.

            FEATURES_TO_ASK:
            {features_to_ask}

            FEATURE DESCRIPTIONS FOR CONTEXT:
            {json.dumps(feature_descriptions, indent=2)}

            RULES:
            - Return ONLY valid JSON (no markdown).
            - Output must be a JSON array of objects.
            - Each object must have keys: "feature" and "question".
            - Each question must be appropriate for the feature type:
              * For numeric features (age, trestbps, chol, thalch, oldpeak, ca): ask for the specific value
              * For categorical features (sex, cp, fbs, restecg, exang, slope, thal): ask yes/no or multiple choice
            - Each question must be ONE sentence and end with '?'.
            - Use clear, patient-friendly language.
            - Every question must be natural and unique.

            Return ONLY the JSON array.
            """

        resp = ollama.chat(
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": prompt}]
        )

        raw = self._clean(resp["message"]["content"])

        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\[\s*{.*}\s*\]", raw, flags=re.S)
            if not m:
                return self._default_heart_questions(features_to_ask)
            try:
                return json.loads(m.group(0))
            except Exception:
                return self._default_heart_questions(features_to_ask)

    def _default_heart_questions(self, features_to_ask: List[str]) -> List[Dict]:
        """Generate default questions for heart features when LLM fails."""
        default_questions = {
            "age": "What is your age?",
            "sex": "What is your biological sex - Male or Female?",
            "cp": "What type of chest pain do you experience - typical angina, atypical angina, non-anginal, or asymptomatic?",
            "trestbps": "What is your resting blood pressure (in mm Hg)?",
            "chol": "What is your serum cholesterol level (in mg/dl)?",
            "fbs": "Is your fasting blood sugar greater than 120 mg/dl?",
            "restecg": "What were your resting ECG results - normal, LV hypertrophy, or ST-T abnormality?",
            "thalch": "What is your maximum heart rate achieved?",
            "exang": "Do you experience chest pain during exercise?",
            "oldpeak": "What is your ST depression value induced by exercise?",
            "slope": "What is the slope of your peak exercise ST segment - upsloping, flat, or downsloping?",
            "ca": "How many major vessels were colored by fluoroscopy (0-3)?",
            "thal": "What is your thalassemia status - normal, fixed defect, or reversable defect?"
        }
        
        return [
            {"feature": f, "question": default_questions.get(f, f"What is your {f.replace('_', ' ')}?")}
            for f in features_to_ask
        ]

    def explain_prediction(self, disease: str, features: Dict, model_type: str = "symptom") -> str:
        """
        Generate a layman-friendly explanation of the prediction.
        
        Args:
            disease: The predicted disease/condition
            features: The features used for prediction
            model_type: Type of model ("symptom" or "heart")
            
        Returns:
            Explanation string
        """
        if model_type == "heart":
            return self._explain_heart_prediction(disease, features)
        else:
            return self._explain_symptom_prediction(disease, features)

    def _explain_symptom_prediction(self, disease: str, features: Dict) -> str:
        """Generate explanation for symptom-based prediction."""
        prompt = f"""
                The predicted disease is {disease}
                Patient symptoms {features}
                Now, Explain in simple language:
                    -What this condition means
                    -Why symptoms match
                    -When to see a doctor

                """
        response = ollama.chat(
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response["message"]["content"]
        text = re.sub(self.THINKING_PATTERN, "", text, flags=re.S).strip()
        
        if "Question:" in text:
            text = text.split("Question:")[-1].strip()
        text = text.replace("**", "").strip()
        return text

    def _explain_heart_prediction(self, prediction: str, features: Dict) -> str:
        """Generate explanation for heart disease prediction."""
        prompt = f"""
                You are a cardiac health educator. A patient has received a preliminary heart disease risk assessment.

                PREDICTION: {prediction}
                PATIENT DATA: {features}

                Explain in simple, non-alarming language:
                1. What this result means (remember, this is a preliminary screening, not a diagnosis)
                2. Which factors in the patient's data may have contributed to this assessment
                3. What the patient should do next (lifestyle changes, see a doctor, etc.)
                4. Important disclaimer that this is NOT a medical diagnosis

                Keep the explanation clear and reassuring. Use bullet points if helpful.
                """
        response = ollama.chat(
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response["message"]["content"]
        text = re.sub(self.THINKING_PATTERN, "", text, flags=re.S).strip()
        text = text.replace("**", "").strip()
        return text

    def get_disclaimer(self, model_type: str = "symptom") -> str:
        """Return appropriate disclaimer based on model type."""
        if model_type == "heart":
            return (
                "WARNING: This heart disease risk assessment is for informational purposes only "
                "and is NOT a medical diagnosis. The results are based on statistical patterns and should not replace "
                "professional medical evaluation. Please consult a cardiologist or your primary care physician for "
                "a comprehensive cardiac assessment. If you experience chest pain, shortness of breath, or other "
                "concerning symptoms, seek immediate medical attention."
            )
        else:
            return (
                "DISCLAIMER: This is not a medical diagnosis. The information provided is for educational "
                "purposes only. Please consult a healthcare professional for proper medical advice. Seek immediate "
                "care if symptoms worsen or you experience severe discomfort."
            )
