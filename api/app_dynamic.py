"""
AI Virtual Clinic - Dynamic Architecture

This is the new Flask application using the dynamic, LLM-driven triage system.
The old model-based predictors have been replaced with an intelligent conversation flow.

Key Changes:
- Removed sklearn model dependencies (symptom, heart, diabetes, mental_health predictors)
- Uses DynamicTriageAgent for intelligent conversation flow
- RAG-powered medical knowledge retrieval
- Multi-modal image support integrated into triage
"""

import os
import sys
import logging

# Add api directory to path for imports FIRST (before any other imports)
api_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(api_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
from api.mongo_db import init_mongo
from api.routes.auth import auth_bp
from api.services.vector_store import VectorStoreFactory
from api.services.embedding_service import EmbeddingServiceFactory
from api.services.rag_service import RAGService
from api.routes.dynamic_triage import create_dynamic_triage_blueprint
from api.routes.booking import create_booking_blueprint
from api.routes.tts import tts_bp
from api.services.medical_knowledge_ingestor import get_medical_knowledge_ingestor
from api.services.clinic_service import get_clinic_service
from api.mongo_db import get_db

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app():
    """
    Application factory for the AI Virtual Clinic API.
    
    New Dynamic Architecture:
    - LLM-driven triage conversation
    - RAG-powered medical knowledge
    - Multi-modal image support
    - Session-based conversation management
    """
    app = Flask(__name__)
    CORS(app)
    
    # JWT Configuration
    app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "super_secret")
    JWTManager(app)
    
    # Initialize MongoDB
    
    init_mongo()
    
    # Register authentication routes
    
    app.register_blueprint(auth_bp)
    
    # Get project paths
    project_root = os.path.normpath(os.path.join(api_dir, ".."))
    
    # ============================================================
    # Initialize RAG Service
    # ============================================================
    rag_service = None
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
                top_k=int(os.environ.get("RAG_TOP_K", "5")),
                min_relevance_score=float(os.environ.get("RAG_MIN_SCORE", "0.3"))
            )
            
            # Index documents if vector store is empty
            if vector_store.count() == 0:
                logger.info("Vector store is empty. Indexing medical knowledge...")
                
                ingestor = get_medical_knowledge_ingestor()
                stats = ingestor.ingest_all_medical_knowledge(project_root)
                logger.info(f"Medical knowledge indexing complete: {stats.to_dict()}")
            
            logger.info(f"RAG service initialized. Documents: {vector_store.count()}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize RAG service: {e}")
            rag_service = None
    else:
        logger.info("RAG service disabled by configuration")
    
    # ============================================================
    # Register Dynamic Triage Routes
    # ============================================================
    dynamic_triage_bp = create_dynamic_triage_blueprint(rag_service=rag_service)
    app.register_blueprint(dynamic_triage_bp)
    
    logger.info("Dynamic triage routes registered at /api/triage")
    
    # ============================================================
    # Register TTS Routes
    # ============================================================
    app.register_blueprint(tts_bp)
    logger.info("TTS routes registered at /api/tts")
    
    # ============================================================
    # Register Booking Routes
    # ============================================================
    
    clinic_service = get_clinic_service()
    db = get_db()
    booking_bp = create_booking_blueprint(clinic_service=clinic_service, db=db)
    app.register_blueprint(booking_bp)
    logger.info("Booking routes registered at /api/booking")
    
    # ============================================================
    # Health Check Endpoint
    # ============================================================
    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "version": "2.0.0",
            "architecture": "dynamic",
            "services": {
                "rag": rag_service is not None and rag_service.is_available() if rag_service else False,
                "mongodb": True
            }
        })
    
    # ============================================================
    # API Info Endpoint
    # ============================================================
    @app.route("/api", methods=["GET"])
    def api_info():
        """API information endpoint."""
        return jsonify({
            "name": "AI Virtual Clinic API",
            "version": "2.0.0",
            "description": "Dynamic, LLM-driven triage system with RAG-powered medical knowledge",
            "endpoints": {
                "triage": {
                    "start": "POST /api/triage/start",
                    "message": "POST /api/triage/message",
                    "image": "POST /api/triage/image",
                    "status": "GET /api/triage/status/<session_id>",
                    "end": "POST /api/triage/end/<session_id>",
                    "history": "GET /api/triage/history/<session_id>",
                    "red_flags": "GET /api/triage/red-flags",
                    "urgency_levels": "GET /api/triage/urgency-levels"
                },
                "booking": {
                    "clinics": "GET /api/booking/clinics",
                    "available_slots": "GET /api/booking/available-slots",
                    "book": "POST /api/booking/book"
                },
                "tts": {
                    "speak": "POST /api/tts/speak"
                },
                "auth": {
                    "login": "POST /api/auth/login",
                    "register": "POST /api/auth/register",
                    "profile": "GET /api/auth/profile"
                }
            }
        })
    
    # ============================================================
    # Error Handlers
    # ============================================================
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "Endpoint not found"
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500
    
    logger.info("AI Virtual Clinic API initialized successfully")
    
    return app


# Create application instance for direct running
app = create_app()


if __name__ == "__main__":
    # Get configuration from environment
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 5000))
    debug = os.environ.get("API_DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting AI Virtual Clinic API on {host}:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=debug
    )
