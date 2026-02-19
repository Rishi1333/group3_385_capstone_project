from flask import Blueprint, request, jsonify
from typing import Dict, Any, Optional

def create_prediction_blueprint(
    symptom_predictor,
    heart_predictor,
    diabetes_predictor,
    mental_health_predictor,
    symptom_disease_extended_predictor,
    symptom_extractor,
    heart_extractor,
    llm,
    store,
    symptom_features,
    heart_features,
    model_router
):
    """
    Create a blueprint that handles predictions for multiple models.
    Routes user input to the appropriate model based on content analysis.
    
    Supported models:
    - symptom: General symptom-based disease prediction
    - heart: Heart disease prediction
    - diabetes: Diabetes prediction
    - mental_health: Mental health screening
    - symptom_disease_extended: NLP-based disease prediction from symptom text
    """
    bp = Blueprint("predictions", __name__)

    @bp.route("/predict/symptoms/start", methods=["POST"])
    def start_conversation():
        """Start a new prediction session - routes to appropriate model."""
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()

        if not text:
            return jsonify({"error": "text field required"}), 400

        # Determine which model to use
        routing_result = model_router.analyze_input(text)
        model_type = routing_result["model_type"]

        # Store session with model type
        session_data = {
            "model_type": model_type,
            "text": text,
            "collected_features": {},
            "routing_info": routing_result
        }
        session_id = store.create(session_data)

        # Route to appropriate handler
        if model_type == "heart":
            return _handle_heart_start(session_id, text, session_data)
        elif model_type == "diabetes":
            return _handle_diabetes_start(session_id, text, session_data)
        elif model_type == "mental_health":
            return _handle_mental_health_start(session_id, text, session_data)
        elif model_type == "symptom_disease":
            return _handle_symptom_disease_extended_start(session_id, text, session_data)
        else:
            return _handle_symptom_start(session_id, text, session_data)

    def _handle_symptom_start(session_id: str, text: str, session_data: Dict) -> tuple:
        """Handle symptom model start."""
        # Extract features from initial text
        extracted = symptom_extractor.extract(text)
        session_data["collected_features"] = extracted
        store.update(session_id, session_data)

        # Determine what we still need to ask
        remaining = [f for f in symptom_features if extracted.get(f, 0) == 0]
        known_positive = [k for k, v in extracted.items() if v == 1]

        # If nothing remaining, we can predict right away
        if not remaining:
            pred = symptom_predictor.predict({"symptoms": extracted})
            explanation = llm.explain_prediction(pred["predictions"], extracted, model_type="symptom")
            disclaimer = llm.get_disclaimer("symptom")
            store.delete(session_id)

            return jsonify({
                "status": "complete",
                "session_id": session_id,
                "model_type": "symptom",
                "final_features": extracted,
                "prediction": pred,
                "explanation": explanation,
                "disclaimer": disclaimer
            }), 200

        # Generate questions for remaining features
        questions = llm.bulk_questions(remaining, known_positive, model_type="symptom")

        return jsonify({
            "status": "questions",
            "session_id": session_id,
            "model_type": "symptom",
            "detected_positive": known_positive,
            "features_extracted": extracted,
            "questions": questions
        }), 200

    def _handle_heart_start(session_id: str, text: str, session_data: Dict) -> tuple:
        """Handle heart model start."""
        # Extract features from initial text
        extracted = heart_extractor.extract(text)
        session_data["collected_features"] = extracted
        store.update(session_id, session_data)

        # Determine what we still need to ask
        remaining = heart_extractor.get_missing_features(extracted)
        known_positive = [k for k, v in extracted.items() if v is not None]

        # If we have enough features, we can predict
        if not remaining:
            pred = heart_predictor.predict({"features": extracted})
            explanation = llm.explain_prediction(pred["predictions"], extracted, model_type="heart")
            disclaimer = llm.get_disclaimer("heart")
            store.delete(session_id)

            return jsonify({
                "status": "complete",
                "session_id": session_id,
                "model_type": "heart",
                "final_features": extracted,
                "prediction": pred,
                "explanation": explanation,
                "disclaimer": disclaimer
            }), 200

        # Generate questions for remaining features
        questions = llm.bulk_questions(remaining, known_positive, model_type="heart")

        return jsonify({
            "status": "questions",
            "session_id": session_id,
            "model_type": "heart",
            "detected_positive": known_positive,
            "features_extracted": extracted,
            "questions": questions
        }), 200

    def _handle_diabetes_start(session_id: str, text: str, session_data: Dict) -> tuple:
        """Handle diabetes model start."""
        # Diabetes model requires specific numeric features
        # We need to ask for them via questions
        diabetes_features = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        
        # Try to extract any features from text (mainly Age)
        extracted = {}
        if "age" in text.lower():
            import re
            age_match = re.search(r'age[:\s]*(\d+)', text.lower())
            if age_match:
                extracted["Age"] = int(age_match.group(1))
        
        session_data["collected_features"] = extracted
        store.update(session_id, session_data)
        
        remaining = [f for f in diabetes_features if f not in extracted]
        
        if not remaining:
            pred = diabetes_predictor.predict({"features": extracted})
            explanation = llm.explain_prediction(pred["prediction"], extracted, model_type="diabetes")
            disclaimer = llm.get_disclaimer("diabetes")
            store.delete(session_id)
            
            return jsonify({
                "status": "complete",
                "session_id": session_id,
                "model_type": "diabetes",
                "final_features": extracted,
                "prediction": pred,
                "explanation": explanation,
                "disclaimer": disclaimer
            }), 200
        
        # Generate questions for remaining features
        questions = llm.bulk_questions(remaining, list(extracted.keys()), model_type="diabetes")
        
        return jsonify({
            "status": "questions",
            "session_id": session_id,
            "model_type": "diabetes",
            "detected_positive": list(extracted.keys()),
            "features_extracted": extracted,
            "questions": questions
        }), 200

    def _handle_mental_health_start(session_id: str, text: str, session_data: Dict) -> tuple:
        """Handle mental health model start."""
        # Mental health model has many categorical features
        # For simplicity, we'll use a questionnaire approach
        mental_health_features = [
            "Age", "Gender", "family_history", "work_interfere", "no_employees",
            "remote_work", "tech_company", "benefits", "care_options", "wellness_program",
            "seek_help", "anonymity", "leave", "mental_health_consequence",
            "phys_health_consequence", "coworkers", "supervisor", "mental_health_interview",
            "phys_health_interview", "mental_vs_physical", "obs_consequence"
        ]
        
        # Extract basic info from text
        extracted = {}
        text_lower = text.lower()
        
        # Try to detect some features from text
        if "family history" in text_lower:
            extracted["family_history"] = "Yes"
        if "work" in text_lower:
            if "stress" in text_lower or "interfere" in text_lower:
                extracted["work_interfere"] = "Often"
        
        session_data["collected_features"] = extracted
        store.update(session_id, session_data)
        
        remaining = [f for f in mental_health_features if f not in extracted]
        
        # For mental health, we typically need many features
        # Show a simplified questionnaire
        questions = llm.bulk_questions(remaining[:5], list(extracted.keys()), model_type="mental_health")
        
        return jsonify({
            "status": "questions",
            "session_id": session_id,
            "model_type": "mental_health",
            "detected_positive": list(extracted.keys()),
            "features_extracted": extracted,
            "questions": questions,
            "note": "Mental health screening requires multiple questions. Answering 5 key questions can provide initial assessment."
        }), 200

    def _handle_symptom_disease_extended_start(session_id: str, text: str, session_data: Dict) -> tuple:
        """Handle NLP-based symptom-disease prediction."""
        # This model takes raw text and predicts disease
        # No feature extraction needed - direct prediction
        pred = symptom_disease_extended_predictor.predict({"text": text})
        explanation = llm.explain_prediction(pred["prediction"], {"text": text}, model_type="symptom_disease_extended")
        disclaimer = llm.get_disclaimer("symptom_disease")
        store.delete(session_id)
        
        return jsonify({
            "status": "complete",
            "session_id": session_id,
            "model_type": "symptom_disease_extended",
            "input_text": text,
            "prediction": pred,
            "explanation": explanation,
            "disclaimer": disclaimer
        }), 200

    @bp.route("/predict/symptoms/submit", methods=["POST"])
    def submit_answers():
        """Submit answers for a prediction session."""
        payload = request.get_json(silent=True) or {}

        session_id = payload.get("session_id")
        answers = payload.get("answers")  # dict feature -> value

        if not session_id:
            return jsonify({"error": "session_id required"}), 400
        if not isinstance(answers, dict):
            return jsonify({"error": "answers must be a dict"}), 400

        session_data = store.get(session_id)
        if not session_data:
            return jsonify({"error": "invalid or expired session"}), 400

        model_type = session_data.get("model_type", "symptom")
        features = session_data["collected_features"]

        # Merge answers into stored features based on model type
        if model_type == "diabetes":
            for feat, val in answers.items():
                try:
                    features[feat] = float(val)
                except (ValueError, TypeError):
                    features[feat] = 0.0
        elif model_type == "mental_health":
            for feat, val in answers.items():
                features[feat] = str(val) if val else "Unknown"
        elif model_type == "heart":
            for feat, val in answers.items():
                if isinstance(val, str):
                    try:
                        features[feat] = float(val)
                    except ValueError:
                        features[feat] = val
                else:
                    features[feat] = val
        else:
            # Symptom model uses binary features
            for feat, val in answers.items():
                if isinstance(val, str):
                    v = val.strip().lower()
                    features[feat] = 1 if v in ("yes", "y", "true", "1") else 0
                else:
                    features[feat] = 1 if int(val) == 1 else 0

        # Predict based on model type
        if model_type == "heart":
            pred = heart_predictor.predict({"features": features})
            explanation = llm.explain_prediction(pred["predictions"], features, model_type="heart")
            disclaimer = llm.get_disclaimer("heart")
        elif model_type == "diabetes":
            pred = diabetes_predictor.predict({"features": features})
            explanation = llm.explain_prediction(pred["prediction"], features, model_type="diabetes")
            disclaimer = llm.get_disclaimer("diabetes")
        elif model_type == "mental_health":
            pred = mental_health_predictor.predict({"features": features})
            explanation = llm.explain_prediction(pred["prediction"], features, model_type="mental_health")
            disclaimer = llm.get_disclaimer("mental_health")
        else:
            pred = symptom_predictor.predict({"symptoms": features})
            explanation = llm.explain_prediction(pred["predictions"], features, model_type="symptom")
            disclaimer = llm.get_disclaimer("symptom")

        # End session
        store.delete(session_id)

        return jsonify({
            "status": "complete",
            "session_id": session_id,
            "model_type": model_type,
            "final_features": features,
            "prediction": pred,
            "explanation": explanation,
            "disclaimer": disclaimer
        }), 200

    @bp.route("/predict/routing", methods=["POST"])
    def analyze_routing():
        """Analyze input text and return routing decision (for debugging/testing)."""
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()

        if not text:
            return jsonify({"error": "text field required"}), 400

        result = model_router.analyze_input(text)
        return jsonify(result), 200

    @bp.route("/predict/direct", methods=["POST"])
    def direct_prediction():
        """
        Direct prediction endpoint that takes model_type and features.
        Bypasses conversation flow for programmatic access.
        """
        payload = request.get_json(silent=True) or {}
        model_type = payload.get("model_type", "symptom_disease")
        features = payload.get("features", {})
        text = payload.get("text", "")
        
        try:
            if model_type == "heart":
                pred = heart_predictor.predict({"features": features})
                disclaimer = llm.get_disclaimer("heart")
            elif model_type == "diabetes":
                pred = diabetes_predictor.predict({"features": features})
                disclaimer = llm.get_disclaimer("diabetes")
            elif model_type == "mental_health":
                pred = mental_health_predictor.predict({"features": features})
                disclaimer = llm.get_disclaimer("mental_health")
            elif model_type == "symptom_disease_extended":
                pred = symptom_disease_extended_predictor.predict({"text": text or str(features)})
                disclaimer = llm.get_disclaimer("symptom_disease")
            else:
                pred = symptom_predictor.predict({"symptoms": features})
                disclaimer = llm.get_disclaimer("symptom")
            
            return jsonify({
                "status": "complete",
                "model_type": model_type,
                "prediction": pred,
                "disclaimer": disclaimer
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @bp.route("/models", methods=["GET"])
    def list_models():
        """List available models and their capabilities."""
        return jsonify({
            "models": [
                {
                    "type": "symptom",
                    "name": "General Symptom-Based Disease Prediction",
                    "description": "Predicts diseases based on binary symptom presence",
                    "features": symptom_features
                },
                {
                    "type": "heart",
                    "name": "Heart Disease Prediction",
                    "description": "Predicts heart disease risk based on clinical features",
                    "features": heart_features if heart_features else ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
                },
                {
                    "type": "diabetes",
                    "name": "Diabetes Prediction",
                    "description": "Predicts diabetes risk based on clinical measurements",
                    "features": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
                },
                {
                    "type": "mental_health",
                    "name": "Mental Health Screening",
                    "description": "Screens for mental health treatment needs based on workplace and personal factors",
                    "features": ["Age", "Gender", "family_history", "work_interfere", "no_employees", "remote_work", "tech_company", "benefits", "care_options", "wellness_program", "seek_help", "anonymity", "leave", "mental_health_consequence", "phys_health_consequence", "coworkers", "supervisor", "mental_health_interview", "phys_health_interview", "mental_vs_physical", "obs_consequence"]
                },
                {
                    "type": "symptom_disease_extended",
                    "name": "NLP-Based Disease Prediction",
                    "description": "Predicts diseases from natural language symptom descriptions using NLP",
                    "features": ["text (natural language description)"]
                }
            ]
        }), 200

    return bp


# Backward compatibility - original symptom_blueprint
def symptom_blueprint(symptom_predictor, feature_extractor, llm, store, all_features):
    """
    Legacy wrapper for backward compatibility.
    Creates a prediction blueprint with only symptom model.
    """
    from typing import Any
    
    # Create dummy predictors that raise errors
    class DummyPredictor:
        def predict(self, *args, **kwargs):
            raise NotImplementedError("Model not configured")
    
    class DummyExtractor:
        def extract(self, text):
            return {}
        def get_missing_features(self, extracted):
            return []
    
    class DummyRouter:
        def analyze_input(self, text):
            return {"model_type": "symptom", "confidence": 1.0}
    
    return create_prediction_blueprint(
        symptom_predictor=symptom_predictor,
        heart_predictor=DummyPredictor(),
        diabetes_predictor=DummyPredictor(),
        mental_health_predictor=DummyPredictor(),
        symptom_disease_extended_predictor=DummyPredictor(),
        symptom_extractor=feature_extractor,
        heart_extractor=DummyExtractor(),
        llm=llm,
        store=store,
        symptom_features=all_features,
        heart_features=[],
        model_router=DummyRouter()
    )
