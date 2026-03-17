import os
import logging
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
from routes.tts import tts_bp
from routes.triage import create_triage_blueprint
from routes.booking import create_booking_blueprint
from services.clinic_service import get_clinic_service

# RAG service imports
from services.vector_store import VectorStore, VectorStoreFactory
from services.embedding_service import EmbeddingService, EmbeddingServiceFactory
from services.rag_service import RAGService, RAGServiceFactory
from services.document_indexer import DocumentIndexer, DocumentIndexerFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    # Initialize RAG service for fallback predictions
    rag_service = None
    rag_available = False
    
    # Check if RAG is enabled
    rag_enabled = os.environ.get("RAG_ENABLED", "true").lower() == "true"
    
    if rag_enabled:
        try:
            logger.info("Initializing RAG service...")
            
            # Initialize vector store
            rag_persist_dir = os.environ.get(
                "RAG_PERSIST_DIR",
                os.path.join(project_root, "chroma_db")
            )
            vector_store = VectorStoreFactory.create(
                backend=os.environ.get("RAG_BACKEND", "chromadb"),
                persist_directory=rag_persist_dir,
                collection_name="medical_knowledge"
            )
            
            # Initialize embedding service
            embedding_service = EmbeddingServiceFactory.get_instance(
                model_name=os.environ.get(
                    "EMBEDDING_MODEL",
                    "sentence-transformers/all-MiniLM-L6-v2"
                ),
                device=os.environ.get("EMBEDDING_DEVICE", "cpu")
            )
            
            # Create RAG service
            rag_service = RAGService(
                vector_store=vector_store,
                embedding_service=embedding_service,
                llm_client=llm,  # Use existing LLM for generation
                top_k=int(os.environ.get("RAG_TOP_K", "5")),
                min_relevance_score=float(os.environ.get("RAG_MIN_SCORE", "0.3"))
            )
            
            # Index documents if vector store is empty
            if vector_store.count() == 0:
                logger.info("Vector store is empty. Indexing documents...")
                indexer = DocumentIndexer(
                    vector_store=vector_store,
                    embedding_service=embedding_service
                )
                indexing_stats = indexer.index_all(project_root, rebuild=False)
                logger.info(f"Document indexing complete: {indexing_stats.to_dict()}")
            
            rag_available = rag_service.is_available()
            logger.info(f"RAG service initialized. Available: {rag_available}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize RAG service: {e}")
            rag_service = None
            rag_available = False
    else:
        logger.info("RAG service disabled by configuration")

    # Initialize model router with RAG service
    model_router = ModelRouter(threshold=0.65, rag_service=rag_service)

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
        model_router=model_router,
        rag_service=rag_service,
        models_available=models_available
    )
    
    app.register_blueprint(bp)
    
    # Register TTS blueprint
    app.register_blueprint(tts_bp)
    
    # Register new Agentic Triage blueprint
    triage_bp = create_triage_blueprint(
        symptom_predictor=symptom_predictor,
        heart_predictor=heart_predictor,
        diabetes_predictor=diabetes_predictor,
        mental_health_predictor=mental_health_predictor,
        rag_service=rag_service
    )
    app.register_blueprint(triage_bp)
    
    # Register Booking blueprint
    from mongo_db import get_db
    clinic_service = get_clinic_service()
    booking_bp = create_booking_blueprint(clinic_service, get_db())
    app.register_blueprint(booking_bp)
    logger.info("Booking service initialized")

    @app.route("/health")
    def health():
        return jsonify({
            "status": "ok",
            "models": models_available,
            "rag": {
                "enabled": rag_enabled,
                "available": rag_available,
                "document_count": rag_service.vector_store.count() if rag_service else 0
            }
        })

    @app.route("/")
    def index():
        return jsonify({
            "name": "AI Virtual Clinic API",
            "version": "2.1.0",
            "description": "Multi-model healthcare prediction API with RAG fallback",
            "models_available": models_available,
            "rag_available": rag_available,
            "endpoints": [
                "/health",
                "/models",
                "/predict/symptoms/start",
                "/predict/symptoms/submit",
                "/predict/routing",
                "/predict/direct",
                "/rag/search",
                "/rag/stats"
            ]
        })
    
    @app.route("/rag/stats")
    def rag_stats():
        """Get RAG service statistics."""
        if not rag_service:
            return jsonify({
                "error": "RAG service not configured"
            }), 404
        
        return jsonify(rag_service.get_stats())
    
    @app.route("/rag/index", methods=["POST"])
    def reindex_documents():
        """Admin endpoint to reindex documents."""
        if not rag_service:
            return jsonify({
                "error": "RAG service not configured"
            }), 404
        
        try:
            indexer = DocumentIndexer(
                vector_store=rag_service.vector_store,
                embedding_service=rag_service.embedding_service
            )
            stats = indexer.index_all(project_root, rebuild=True)
            return jsonify({
                "status": "success",
                "stats": stats.to_dict()
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=3000, debug=True)


