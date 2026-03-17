"""
Dynamic Triage Routes

API endpoints for the new Dynamic Triage Workflow.
Replaces the old model-centric triage with LLM-driven conversation.
"""

import logging
import os
import sys
import json
from flask import Blueprint, request, jsonify
from io import BytesIO
from datetime import datetime
from flask_jwt_extended import verify_jwt_in_request, get_jwt
from werkzeug.utils import secure_filename

# Add api directory to path for imports
api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

# Services
from api.services.dynamic_triage_agent import DynamicTriageAgent, TriageState, create_dynamic_triage_agent
from api.services.vision_processor import get_vision_processor
from api.services.clinical_report import get_clinical_report_generator

# Handle import for both running from project root and from api directory
try:
    from api.mongo_db import get_db
except ImportError:
    from mongo_db import get_db

# Configure logging
logger = logging.getLogger(__name__)


# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_dynamic_triage_blueprint(rag_service=None) -> Blueprint:
    """
    Create the dynamic triage blueprint.
    
    Args:
        rag_service: Optional RAG service for medical knowledge
        
    Returns:
        Flask blueprint
    """
    bp = Blueprint("dynamic_triage", __name__, url_prefix="/api/triage")
    
    # Vision processor for image analysis
    vision_processor = get_vision_processor()
    
    # Clinical report generator
    report_generator = get_clinical_report_generator()
    
    # Simple in-memory agent store (bypassing SessionStore complexity)
    _agents = {}
    
    def _get_or_create_agent(session_id: str) -> DynamicTriageAgent:
        """Get existing agent or create new one."""
        if session_id in _agents:
            return _agents[session_id]
        
        # Create new agent
        agent = create_dynamic_triage_agent(
            session_id=session_id,
            rag_service=rag_service,
            vision_processor=vision_processor
        )
        
        _agents[session_id] = agent
        return agent
    
    def _persist_to_patient_db(session_id: str, clinical_data: dict, report: dict = None, formatted_report: str = None) -> None:
        """Persist clinical data to patient database if JWT present."""
        try:
            verify_jwt_in_request(optional=True)
            claims = get_jwt() or {}
        except Exception:
            return
        
        if claims.get("role") != "patients":
            return
        
        email = claims.get("email")
        if not email:
            return
        
        try:
            db = get_db()
            
            # Build clinical snapshot
            profile = clinical_data.get("patient_profile", {}) or {}
            symptoms = [s.get("name") if isinstance(s, dict) else s 
                       for s in clinical_data.get("symptoms", [])]
            
            clinical_snapshot = {
                "age": profile.get("age"),
                "sex": profile.get("sex"),
                "chief_complaint": clinical_data.get("chief_complaint"),
                "symptoms": symptoms,
                "medical_history": clinical_data.get("medical_history", []),
                "medications": clinical_data.get("medications", []),
                "urgency_level": clinical_data.get("urgency_level"),
                "session_id": session_id,
                "stored_at": datetime.utcnow().isoformat()
            }
            
            update_data = {
                "$set": {
                    "updated_at": datetime.utcnow().isoformat(),
                    "last_clinical_snapshot": clinical_snapshot
                }
            }
            
            if report:
                # Add formatted_report and uploaded_image to the report before storing
                report_to_store = report.copy()
                report_to_store["formatted_report"] = formatted_report
                # Include uploaded image if available
                if agent.clinical_data.get("uploaded_image_base64"):
                    report_to_store["uploaded_image"] = agent.clinical_data["uploaded_image_base64"]
                update_data["$push"] = {
                    "clinical_reports": {
                        "$each": [report_to_store],
                        "$position": 0
                    }
                }
            
            db["patients"].update_one(
                {"email": email},
                update_data
            )
            
        except Exception as exc:
            logger.warning(f"Could not persist to patient DB: {exc}")
    
    @bp.route("/start", methods=["POST", "GET"])
    def start_session():
        """Start a new triage session."""
        try:
            # Handle both JSON and empty requests
            try:
                data = request.get_json(force=True, silent=True) or {}
            except:
                data = {}
            session_id = data.get("session_id") or os.urandom(16).hex()
            
            # Create new agent
            agent = create_dynamic_triage_agent(
                session_id=session_id,
                rag_service=rag_service,
                vision_processor=vision_processor
            )
            
            # Store in our simple agent store
            _agents[session_id] = agent
            
            return jsonify({
                "success": True,
                "session_id": session_id,
                "state": agent.get_state(),
                "message": "Hello! I'm here to help you with your health concerns. What brings you in today?"
            })
            
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @bp.route("/message", methods=["POST"])
    def process_message():
        """Process a user message in the triage conversation."""
        try:
            data = request.get_json() or {}
            session_id = data.get("session_id")
            user_message = data.get("message", "")
            
            if not session_id:
                return jsonify({
                    "success": False,
                    "error": "session_id is required"
                }), 400
            
            # Allow empty message if it's just a continuation after image upload
            # but still require session_id
            if not user_message and not request.files:
                return jsonify({
                    "success": False,
                    "error": "message is required"
                }), 400
            
            # Get or create agent
            agent = _get_or_create_agent(session_id)
            
            # Process the message
            response = agent.process_input(user_message)
            
            # Check if session is complete
            if response.get("end_session"):
                # Generate clinical report
                clinical_data = agent.get_clinical_data()
                
                # Extract image findings if available
                image_findings = clinical_data.get("image_findings")
                
                report = report_generator.generate_report(
                    clinical_data=clinical_data,
                    tool_results=None,
                    image_findings=image_findings
                )
                
                # Format report for display (markdown with styling)
                formatted_report = report_generator.format_for_display(report)
                
                response["report"] = report
                response["formatted_report"] = formatted_report
                
                # Persist to database with formatted_report
                _persist_to_patient_db(session_id, clinical_data, report, formatted_report)
                
                # Clear session from our simple store
                if session_id in _agents:
                    del _agents[session_id]
            
            return jsonify({
                "success": True,
                **response
            })
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @bp.route("/image", methods=["POST"])
    def upload_image():
        """Upload and process a medical image."""
        try:
            session_id = request.form.get("session_id")
            
            if not session_id:
                return jsonify({
                    "success": False,
                    "error": "session_id is required"
                }), 400
            
            if 'image' not in request.files:
                return jsonify({
                    "success": False,
                    "error": "No image file provided"
                }), 400
            
            file = request.files['image']
            
            if file.filename == '':
                return jsonify({
                    "success": False,
                    "error": "No image file selected"
                }), 400
            
            if not allowed_file(file.filename):
                return jsonify({
                    "success": False,
                    "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                }), 400
            
            # Get agent
            agent = _get_or_create_agent(session_id)
            
            # Read image data
            image_data = file.read()
            
            # Encode to base64 for storage
            import base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Process with agent
            response = agent.process_input("", image_data=image_data)
            
            # Store image base64 in response for display
            if image_base64:
                response["uploaded_image"] = image_base64
                # Also store in clinical_data for the report
                if agent.clinical_data.get("image_findings"):
                    agent.clinical_data["uploaded_image_base64"] = image_base64
            
            return jsonify({
                "success": True,
                **response
            })
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @bp.route("/status/<session_id>", methods=["GET"])
    def get_status(session_id: str):
        """Get the current status of a triage session."""
        try:
            if session_id not in _agents:
                return jsonify({
                    "success": False,
                    "error": "Session not found"
                }), 404
            
            agent = _agents[session_id]
            
            return jsonify({
                "success": True,
                "session_id": session_id,
                "state": agent.get_state(),
                "clinical_data": agent.get_clinical_data(),
                "questions_asked": agent.questions_asked
            })
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @bp.route("/end/<session_id>", methods=["POST"])
    def end_session(session_id: str):
        """End a triage session and generate summary."""
        try:
            if session_id not in _agents:
                return jsonify({
                    "success": False,
                    "error": "Session not found"
                }), 404
            
            agent = _agents[session_id]
            
            # Force transition to summary
            agent.state = TriageState.CLINICAL_SUMMARY
            response = agent.process_input("Please generate my summary.")
            
            # Generate report
            clinical_data = agent.get_clinical_data()
            
            # Extract image findings if available
            image_findings = clinical_data.get("image_findings")
            
            report = report_generator.generate_report(
                clinical_data=clinical_data,
                tool_results=None,
                image_findings=image_findings
            )
            
            # Format report for display
            formatted_report = report_generator.format_for_display(report)
            
            # Persist to database with formatted_report
            _persist_to_patient_db(session_id, clinical_data, report, formatted_report)
            
            # Clear session
            del _agents[session_id]
            
            return jsonify({
                "success": True,
                "message": response.get("message"),
                "report": report,
                "formatted_report": formatted_report,
                "clinical_data": clinical_data
            })
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @bp.route("/history/<session_id>", methods=["GET"])
    def get_history(session_id: str):
        """Get conversation history for a session."""
        try:
            if session_id not in _agents:
                return jsonify({
                    "success": False,
                    "error": "Session not found"
                }), 404
            
            agent = _agents[session_id]
            
            return jsonify({
                "success": True,
                "session_id": session_id,
                "history": agent.get_message_history()
            })
            
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @bp.route("/red-flags", methods=["GET"])
    def get_red_flags():
        """Get red flag definitions for reference."""
        try:
            # Load red flags from config
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', 'red_flags.json'
            )
            
            with open(config_path, 'r') as f:
                red_flags = json.load(f)
            
            return jsonify({
                "success": True,
                "red_flags": red_flags
            })
            
        except Exception as e:
            logger.error(f"Error loading red flags: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @bp.route("/urgency-levels", methods=["GET"])
    def get_urgency_levels():
        """Get urgency level definitions."""
        return jsonify({
            "success": True,
            "urgency_levels": {
                "CRITICAL": {
                    "description": "Life-threatening, requires immediate emergency services",
                    "response_time": "Call 911 immediately",
                    "color": "red"
                },
                "HIGH": {
                    "description": "Urgent, requires care within hours",
                    "response_time": "Go to ER or urgent care within 2-4 hours",
                    "color": "orange"
                },
                "MODERATE": {
                    "description": "Should see doctor within 24-48 hours",
                    "response_time": "Schedule appointment within 1-2 days",
                    "color": "yellow"
                },
                "LOW": {
                    "description": "Routine, can wait for regular appointment",
                    "response_time": "Schedule routine appointment",
                    "color": "green"
                }
            }
        })
    
    return bp
