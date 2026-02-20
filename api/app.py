import os
from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager

from db import db
from routes.auth import auth_bp

from services.artifact_loader import ArtifactLoader
from services.feature_extractor import FeatureExtractorFactory
from services.symptom_predictor import SymptomPredictor
from services.heart_predictor import HeartPredictor
from services.diabetes_predictor import DiabetesPredictor
from services.mental_health_predictor import MentalHealthPredictor
from services.symptom_disease_extended_predictor import SymptomDiseaseExtendedPredictor
from services.llm_conversation import LLMConversation
from services.session_store import SessionStore
from services.model_router import ModelRouter
from routes.symptoms import create_prediction_blueprint


def create_app():
    app = Flask(__name__)
    CORS(app)

    
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///clinic.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


    jwt_secret = os.environ.get("JWT_SECRET_KEY")
    if not jwt_secret:
        raise RuntimeError("JWT_SECRET_KEY is not set. Set it as an environment variable.")

    app.config["JWT_SECRET_KEY"] = jwt_secret

    # Init extensions
    db.init_app(app)
    JWTManager(app)

    # Create tables (first run)
    with app.app_context():
        db.create_all()

    
    ARTIFACT_PATH = os.environ.get("ARTIFACT_PATH", r"your_path_to_artifacts")
    SYMPTOM_KEYWORDS_FILE = os.environ.get("SYMPTOM_KEYWORDS_FILE", r"your_path_to_config/symptom_mapping.json")
    HEART_KEYWORDS_FILE = os.environ.get("HEART_KEYWORDS_FILE", r"your_path_to_config/heart_feature_mapping.json")

    # Load model artifacts
    loader = ArtifactLoader(ARTIFACT_PATH)

    models_available = {
        "symptom": False,
        "heart": False,
        "diabetes": False,
        "mental_health": False,
        "symptom_disease_extended": False
    }

    # ----------------------------
    # REQUIRED: Symptom model
    # ----------------------------
    try:
        symptom_bundle = loader.load_symptom_bundle()
        symptom_model = symptom_bundle["model"]
        symptom_config = symptom_bundle["config"]
        symptom_predictor = SymptomPredictor(symptom_model, symptom_config)
        symptom_features = symptom_config["features"]
        models_available["symptom"] = True
    except FileNotFoundError as e:
        print(f"Error: Symptom model not found: {e}")
        raise RuntimeError("Symptom model is required but not found")

    # ----------------------------
    # Heart model
    # ----------------------------
    try:
        heart_bundle = loader.load_heart_bundle()
        heart_model = heart_bundle["model"]
        heart_config = heart_bundle["config"]
        heart_predictor = HeartPredictor(heart_model, heart_config)
        heart_features = heart_config["features"]
        models_available["heart"] = True
        print("Heart model loaded successfully")
    except FileNotFoundError:
        print("Warning: Heart model not found. Heart predictions will be unavailable.")
        heart_predictor = None
        heart_config = None
        heart_features = []

    # ----------------------------
    # Diabetes model
    # ----------------------------
    try:
        diabetes_bundle = loader.load_diabetes_bundle()
        diabetes_model = diabetes_bundle["model"]
        diabetes_config = diabetes_bundle["config"]
        diabetes_predictor = DiabetesPredictor(diabetes_model, diabetes_config)
        models_available["diabetes"] = True
        print("Diabetes model loaded successfully")
    except FileNotFoundError:
        print("Warning: Diabetes model not found. Diabetes predictions will be unavailable.")
        diabetes_predictor = None
        diabetes_config = None

    # ----------------------------
    # Mental health model
    # ----------------------------
    try:
        mental_health_bundle = loader.load_mental_health_bundle()
        mental_health_model = mental_health_bundle["model"]
        mental_health_config = mental_health_bundle["config"]
        mental_health_predictor = MentalHealthPredictor(mental_health_model, mental_health_config)
        models_available["mental_health"] = True
        print("Mental health model loaded successfully")
    except FileNotFoundError:
        print("Warning: Mental health model not found. Mental health screening will be unavailable.")
        mental_health_predictor = None
        mental_health_config = None

    # ----------------------------
    # Symptom-disease extended (NLP)
    # ----------------------------
    try:
        sde_bundle = loader.load_symptom_disease_extended_bundle()
        symptom_disease_extended_predictor = SymptomDiseaseExtendedPredictor(
            model=sde_bundle["model"],
            config=sde_bundle["config"],
            label_encoder=sde_bundle.get("label_encoder")
        )
        models_available["symptom_disease_extended"] = True
        print("Symptom-disease extended (NLP) model loaded successfully")
    except FileNotFoundError:
        print("Warning: Symptom-disease extended model not found. NLP predictions will be unavailable.")
        symptom_disease_extended_predictor = None

    # Feature extractors
    symptom_extractor = FeatureExtractorFactory.create("symptom", SYMPTOM_KEYWORDS_FILE, symptom_config)
    heart_extractor = (
        FeatureExtractorFactory.create("heart", HEART_KEYWORDS_FILE, heart_config)
        if models_available["heart"]
        else None
    )

    # LLM + sessions + router
    llm = LLMConversation()
    store = SessionStore(ttl_seconds=900)
    model_router = ModelRouter(threshold=0.65)

    # Dummy predictor for missing models
    class DummyPredictor:
        def predict(self, *args, **kwargs):
            raise NotImplementedError("Model not configured")

    # Prediction blueprint
    bp = create_prediction_blueprint(
        symptom_predictor=symptom_predictor,
        heart_predictor=heart_predictor or DummyPredictor(),
        diabetes_predictor=diabetes_predictor or DummyPredictor(),
        mental_health_predictor=mental_health_predictor or DummyPredictor(),
        symptom_disease_extended_predictor=symptom_disease_extended_predictor or DummyPredictor(),
        symptom_extractor=symptom_extractor,
        heart_extractor=heart_extractor,
        llm=llm,
        store=store,
        symptom_features=symptom_features,
        heart_features=heart_features,
        model_router=model_router
    )
    app.register_blueprint(bp)

    # Auth routes
    app.register_blueprint(auth_bp, url_prefix="/auth")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "models": models_available})

    @app.route("/")
    def index():
        return jsonify({
            "name": "Virtual Clinic API",
            "version": "2.0.0",
            "models_available": models_available
        })

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=3000, debug=True)
