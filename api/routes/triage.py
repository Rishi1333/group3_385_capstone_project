"""
Triage Routes

API endpoints for the Agentic Triage Workflow.
"""

import logging
from flask import Blueprint, request, jsonify
from io import BytesIO
from datetime import datetime
from flask_jwt_extended import verify_jwt_in_request, get_jwt
from mongo_db import get_db

# Services (will be injected)
from services.safety_guardrails import get_safety_guardrails
from services.semantic_router import get_semantic_router
from services.tool_registry import get_tool_registry
from services.triage_agent import TriageAgent, TriageState
from services.clinical_report import get_clinical_report_generator
from services.vision_processor import get_vision_processor
from services.session_store import get_session_store

# Configure logging
logger = logging.getLogger(__name__)


def create_triage_blueprint(
    symptom_predictor=None,
    heart_predictor=None,
    diabetes_predictor=None,
    mental_health_predictor=None,
    rag_service=None
):
    """
    Create the triage blueprint with all services.
    
    Args:
        symptom_predictor: Symptom prediction model
        heart_predictor: Heart prediction model
        diabetes_predictor: Diabetes prediction model
        mental_health_predictor: Mental health prediction model
        rag_service: RAG service for general queries
        
    Returns:
        Flask blueprint
    """
    bp = Blueprint("triage", __name__, url_prefix="/api/triage")
    
    # Store references to models for tool invocation
    models = {
        "symptom": symptom_predictor,
        "heart": heart_predictor,
        "diabetes": diabetes_predictor,
        "mental_health": mental_health_predictor
    }
    
    # Initialize services lazily
    def get_services():
        return {
            "safety": get_safety_guardrails(),
            "router": get_semantic_router(),
            "tool_registry": get_tool_registry(),
            "report_generator": get_clinical_report_generator(),
            "vision": get_vision_processor(),
            "store": get_session_store()
        }

    def _report_for_db(report: dict) -> dict:
        """Return a report payload safe for database display (no disclaimer)."""
        clean = dict(report or {})
        clean.pop("disclaimer", None)
        return clean

    def _build_clinical_snapshot(clinical_data: dict) -> dict:
        """Extract key clinical fields for patient database storage."""
        data = clinical_data or {}
        profile = data.get("patient_profile", {}) or {}

        symptoms = []
        for item in data.get("symptoms", []):
            if isinstance(item, dict):
                symptoms.append(item.get("name"))
            else:
                symptoms.append(item)

        # Remove empty symptom values
        symptoms = [s for s in symptoms if s]

        return {
            "age": profile.get("age"),
            "sex": profile.get("sex"),
            "chief_complaint": data.get("chief_complaint"),
            "symptoms": symptoms,
            "medical_history": data.get("medical_history", []),
            "medications": data.get("medications", []),
            "risk_factors": data.get("risk_factors", []),
            "image_findings": data.get("image_findings")
        }

    def _persist_report_to_patient_db(report: dict, clinical_data: dict = None) -> None:
        """
        Persist report to the current patient document if JWT is present.
        This keeps database records without disclaimer text.
        """
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

        db_report = _report_for_db(report)
        db_report["stored_at"] = datetime.utcnow().isoformat()
        clinical_snapshot = _build_clinical_snapshot(clinical_data)
        db_report["clinical_snapshot"] = clinical_snapshot

        try:
            db = get_db()
            db["patients"].update_one(
                {"email": email},
                {
                    "$push": {"clinical_reports": {"$each": [db_report], "$position": 0}},
                    "$set": {
                        "updated_at": datetime.utcnow().isoformat(),
                        "last_clinical_snapshot": clinical_snapshot,
                        "last_report_id": db_report.get("report_id"),
                        "last_report_generated_at": db_report.get("generated_at")
                    }
                }
            )
        except Exception as exc:
            logger.warning(f"Could not persist report to patient DB: {exc}")
    
    # Register callables with tool registry
    tool_registry = get_tool_registry()
    
    if symptom_predictor:
        tool_registry.register_callable("symptom_predict", 
            lambda data: symptom_predictor.predict(data))
    if heart_predictor:
        tool_registry.register_callable("heart_analysis",
            lambda data: heart_predictor.predict(data))
    if diabetes_predictor:
        tool_registry.register_callable("diabetes_risk",
            lambda data: diabetes_predictor.predict(data))
    if mental_health_predictor:
        tool_registry.register_callable("mental_health_screen",
            lambda data: mental_health_predictor.predict(data))
    
    @bp.route("/start", methods=["POST"])
    def start_triage():
        """
        Start a new triage session.
        
        Request body:
            {
                "text": "Patient's initial complaint",
                "image": (optional) base64 encoded image
            }
            
        Returns:
            {
                "session_id": "uuid",
                "state": "GATHERING",
                "question": "First question to ask"
            }
        """
        services = get_services()
        
        # Handle both JSON and multipart/form-data
        content_type = request.content_type or ""
        if "multipart/form-data" in content_type:
            text = (request.form.get("text") or "").strip()
            image_file = request.files.get("image")
        else:
            payload = request.get_json(silent=True) or {}
            text = (payload.get("text") or "").strip()
            image_file = None
        
        if not text:
            return jsonify({"error": "text field required"}), 400
        
        # Step 1: Safety check
        safety_result = services["safety"].detect_emergency(text)
        
        if safety_result["is_emergency"]:
            return jsonify({
                "status": "emergency",
                "emergency_response": services["safety"].get_emergency_response(),
                "severity": safety_result["severity"],
                "recommended_action": safety_result["recommended_action"]
            }), 200
        
        # Step 2: Intent classification
        has_image = image_file is not None
        intent_result = services["router"].classify_intent(text, has_image)
        
        # Handle emergency intent
        if intent_result["intent"] == "EMERGENCY":
            return jsonify({
                "status": "emergency",
                "emergency_response": services["safety"].get_emergency_response(),
                "reasoning": intent_result.get("reasoning")
            }), 200
        
        # Handle general query with RAG
        if intent_result["intent"] == "GENERAL_QUERY":
            if rag_service and rag_service.is_available():
                try:
                    rag_response = rag_service.predict_with_rag(
                        symptoms=[text],
                        features={},
                        model_type="symptom"
                    )
                    return jsonify({
                        "status": "complete",
                        "intent": "GENERAL_QUERY",
                        "response": rag_response.to_dict(),
                        "disclaimer": rag_response.disclaimer
                    }), 200
                except Exception as e:
                    logger.error(f"RAG error: {e}")
            
            return jsonify({
                "status": "complete",
                "intent": "GENERAL_QUERY",
                "response": "I'm here to help with symptom assessment. Please describe your symptoms.",
                "disclaimer": "This is AI assistance, not medical advice."
            }), 200
        
        # Start triage for symptom triage intent
        # Create session
        session_id = services["store"].create({
            "intent": intent_result["intent"],
            "original_text": text
        })
        
        # Create triage agent
        agent = TriageAgent(
            session_id=session_id,
            tool_registry=services["tool_registry"]
        )
        
        # Start intake
        result = agent.start_intake(text)
        
        # Save to session
        services["store"].update(session_id, agent.to_dict())
        
        # Add message to history
        services["store"].add_message(session_id, "assistant", result.get("question", ""))
        
        return jsonify({
            "session_id": session_id,
            "state": result.get("state", "GATHERING"),
            "question": result.get("question"),
            "intent": intent_result["intent"],
            "safety_check": safety_result
        }), 200
    
    @bp.route("/message", methods=["POST"])
    def send_message():
        """
        Send a message in an existing triage session.
        
        Request body:
            {
                "session_id": "uuid",
                "message": "User's response"
            }
            
        Returns:
            {
                "session_id": "uuid",
                "state": "GATHERING|ANALYZING|REPORTING|COMPLETE",
                "question": "Next question (if gathering)",
                "report": {...} (if complete)
            }
        """
        services = get_services()
        
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id")
        message = (payload.get("message") or "").strip()
        
        if not session_id:
            return jsonify({"error": "session_id required"}), 400
        if not message:
            return jsonify({"error": "message required"}), 400
        
        # Get session
        session_data = services["store"].get(session_id)
        if not session_data:
            return jsonify({"error": "Invalid or expired session"}), 400
        
        # Get current state (stored as "state" in to_dict())
        current_state = session_data.get("state", "IDLE")
        
        # Restore agent
        agent = TriageAgent.from_dict(
            session_data,
            services["tool_registry"]
        )
        
        # Process based on state
        if current_state == "GATHERING":
            result = agent.process_message(message)
            services["store"].update(session_id, agent.to_dict())
            services["store"].add_message(session_id, "user", message)
            services["store"].add_message(session_id, "assistant", result.get("question", ""))
            
            # Check if transitioned to analyzing
            if result.get("state") == "ANALYZING":
                clinical_data = agent.get_clinical_data()

                # Invoke tools
                tool_results = []
                for tool_name in result.get("tools_to_invoke", []):
                    # Prepare tool data
                    tool_data = {
                        "symptoms": {s["name"]: 1 for s in clinical_data.get("symptoms", [])},
                        "age": clinical_data.get("patient_profile", {}).get("age"),
                        "sex": clinical_data.get("patient_profile", {}).get("sex")
                    }
                    
                    # Invoke tool
                    tool_result = services["tool_registry"].invoke_tool(tool_name, tool_data)
                    tool_results.append({
                        "tool": tool_name,
                        "result": tool_result
                    })
                
                # Generate report
                services["store"].update(session_id, agent.to_dict())
                
                report_result = services["report_generator"].generate_report(
                    clinical_data=clinical_data,
                    tool_results=tool_results,
                    image_findings=clinical_data.get("image_findings")
                )
                
                # Save report
                services["store"].set_report(session_id, report_result)
                _persist_report_to_patient_db(report_result, clinical_data)
                
                # Format for display
                formatted_report = services["report_generator"].format_for_display(report_result)
                
                # Extract conditions for booking recommendations
                conditions = []
                for diff in report_result.get("suggested_differential", []):
                    cond = diff.get("condition")
                    if isinstance(cond, list):
                        conditions.extend(cond)
                    elif cond:
                        conditions.append(cond)
                
                return jsonify({
                    "session_id": session_id,
                    "state": "COMPLETE",
                    "report": report_result,
                    "report_display": formatted_report,
                    "conditions": conditions,  # For booking recommendations
                    "disclaimer": report_result.get("disclaimer")
                }), 200
            
            return jsonify({
                "session_id": session_id,
                "state": result.get("state", "GATHERING"),
                "question": result.get("question"),
                "clinical_data_summary": agent.get_clinical_data()
            }), 200
        
        elif current_state == "ANALYZING":
            # Just acknowledge
            return jsonify({
                "session_id": session_id,
                "state": current_state,
                "message": "Processing your information..."
            }), 200
        
        else:
            return jsonify({
                "error": f"Cannot process message in {current_state} state"
            }), 400
    
    @bp.route("/image", methods=["POST"])
    def upload_image():
        """
        Upload and analyze a medical image.
        
        Request: multipart/form-data
            - session_id: (optional) existing session
            - image: image file
            
        Returns:
            {
                "session_id": "uuid",
                "analysis": "Image analysis results"
            }
        """
        services = get_services()
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({"error": "image file required"}), 400
        
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Validate image
        validation = services["vision"].validate_image(image_data)
        if not validation["valid"]:
            return jsonify({"error": validation["error"]}), 400
        
        # Process image
        context = request.form.get("context", "")
        result = services["vision"].process_image(image_data, context)
        
        if not result["success"]:
            return jsonify({"error": result.get("error", "Image processing failed")}), 500
        
        # Get or create session
        session_id = request.form.get("session_id")
        
        if session_id:
            session_data = services["store"].get(session_id)
            if session_data:
                # Update existing session
                agent = TriageAgent.from_dict(session_data, services["tool_registry"])
                agent.set_image_findings(result["analysis"])
                services["store"].update(session_id, agent.to_dict())
        else:
            # Create new session
            session_id = services["store"].create({
                "intent": "IMAGE_ANALYSIS",
                "image_analysis": result["analysis"]
            })
        
        return jsonify({
            "session_id": session_id,
            "analysis": result["analysis"],
            "image_valid": validation
        }), 200
    
    @bp.route("/<session_id>", methods=["GET"])
    def get_session(session_id):
        """
        Get session status.
        
        Returns:
            Session data
        """
        services = get_services()
        
        session_data = services["store"].get(session_id)
        if not session_data:
            return jsonify({"error": "Session not found"}), 404
        
        return jsonify({
            "session_id": session_id,
            "state": session_data.get("current_state"),
            "intent": session_data.get("intent"),
            "created_at": session_data.get("created_at"),
            "clinical_data": session_data.get("clinical_data")
        }), 200
    
    @bp.route("/<session_id>/report", methods=["GET"])
    def get_report(session_id):
        """
        Get the final clinical report.
        
        Returns:
            Clinical report
        """
        services = get_services()
        
        session_data = services["store"].get(session_id)
        if not session_data:
            return jsonify({"error": "Session not found"}), 404
        
        report = session_data.get("final_report")
        if not report:
            return jsonify({"error": "Report not available"}), 404
        
        # Format for display
        formatted = services["report_generator"].format_for_display(report)
        
        return jsonify({
            "report": report,
            "formatted": formatted
        }), 200
    
    @bp.route("/<session_id>", methods=["DELETE"])
    def delete_session(session_id):
        """
        Delete a session and all associated data.
        
        Returns:
            Success message
        """
        services = get_services()
        
        success = services["store"].delete(session_id)
        if not success:
            return jsonify({"error": "Session not found"}), 404
        
        return jsonify({
            "message": "Session deleted successfully",
            "session_id": session_id
        }), 200
    
    @bp.route("/routing", methods=["POST"])
    def analyze_routing():
        """
        Analyze input for routing (debug/testing endpoint).
        
        Request:
            {
                "text": "User input"
            }
            
        Returns:
            Routing analysis
        """
        services = get_services()
        
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        
        if not text:
            return jsonify({"error": "text field required"}), 400
        
        # Safety check
        safety = services["safety"].detect_emergency(text)
        
        # Intent classification
        intent = services["router"].classify_intent(text)
        
        return jsonify({
            "safety_check": safety,
            "intent_classification": intent
        }), 200
    
    return bp
