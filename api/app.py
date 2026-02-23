import os
from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from routes.auth import auth_bp
from dotenv import load_dotenv
load_dotenv()
from mongo_db import init_mongo
from routes.auth import auth_bp
from flask_jwt_extended import JWTManager

from services.artifact_loader import ArtifactLoader
from services.feature_extractor import FeatureExtractorFactory, SymptomFeatureExtractor, HeartFeatureExtractor
from services.symptom_predictor import SymptomPredictor
from services.heart_predictor import HeartPredictor
from services.diabetes_predictor import DiabetesPredictor
from services.mental_health_predictor import MentalHealthPredictor
from services.llm_conversation import LLMConversation
from services.session_store import SessionStore
from services.model_router import ModelRouter
from routes.symptoms import create_prediction_blueprint, symptom_blueprint

def create_app():
    """
    Application factory for the AI Virtual Clinic API.
    Loads models, initializes services, and registers routes.
    
    Supported models:
    - symptom: General symptom-based disease prediction
    - heart: Heart disease prediction
    - diabetes: Diabetes prediction
    - mental_health: Mental health screening
    """
    app = Flask(__name__)
    CORS(app)
# JWT
    app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "super_secret")
    JWTManager(app)
    
    # Mongo
    init_mongo()
    
    # Register auth routes
    app.register_blueprint(auth_bp)


    api_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(api_dir, ".."))
    
    # Configuration paths - use environment variables or defaults
    ARTIFACT_PATH = os.environ.get(
        "ARTIFACT_PATH",
        os.path.join(project_root, "artifacts")
    )
    SYMPTOM_KEYWORDS_FILE = os.environ.get(
        "SYMPTOM_KEYWORDS_FILE",
        os.path.join(api_dir, "config", "symptom_mapping.json")
    )
    HEART_KEYWORDS_FILE = os.environ.get(
        "HEART_KEYWORDS_FILE",
        os.path.join(api_dir, "config", "heart_feature_mapping.json")
    )

    # Load model artifacts
    loader = ArtifactLoader(ARTIFACT_PATH)
    
    # Track model availability
    models_available = {
        "symptom": False,
        "heart": False,
        "diabetes": False,
        "mental_health": False
    }
    
    # Load symptom model (required)
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
    
    # Load heart model (optional)
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

    # Load diabetes model (optional)
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

    # Load mental health model (optional)
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

    # Initialize feature extractors
    symptom_extractor = FeatureExtractorFactory.create(
        "symptom",
        SYMPTOM_KEYWORDS_FILE,
        symptom_config
    )
    
    if models_available["heart"]:
        heart_extractor = FeatureExtractorFactory.create(
            "heart",
            HEART_KEYWORDS_FILE,
            heart_config
        )
    else:
        heart_extractor = None

    # Initialize LLM and session store
    llm = LLMConversation()
    store = SessionStore(ttl_seconds=900)

    # Initialize model router
    model_router = ModelRouter(threshold=0.65)

    # Create dummy predictors for unavailable models
    class DummyPredictor:
        def predict(self, *args, **kwargs):
            raise NotImplementedError("Model not configured")
    
    # Register prediction blueprint with all models
    bp = create_prediction_blueprint(
        symptom_predictor=symptom_predictor,
        heart_predictor=heart_predictor or DummyPredictor(),
        diabetes_predictor=diabetes_predictor or DummyPredictor(),
        mental_health_predictor=mental_health_predictor or DummyPredictor(),
        symptom_extractor=symptom_extractor,
        heart_extractor=heart_extractor,
        llm=llm,
        store=store,
        symptom_features=symptom_features,
        heart_features=heart_features,
        model_router=model_router
    )
    
    app.register_blueprint(bp)

    @app.route("/health")
    def health():
        return jsonify({
            "status": "ok",
            "models": models_available
        })

    @app.route("/")
    def index():
        return jsonify({
            "name": "AI Virtual Clinic API",
            "version": "2.0.0",
            "description": "Multi-model healthcare prediction API",
            "models_available": models_available,
            "endpoints": [
                "/health",
                "/models",
                "/predict/symptoms/start",
                "/predict/symptoms/submit",
                "/predict/routing",
                "/predict/direct"
            ]
        })
    

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=3000, debug=True)


