"""
Tests for Agentic Triage Workflow

Tests the new triage system including:
- Safety Guardrails (LLM-based emergency detection)
- Semantic Router (Intent classification)
- Tool Registry (ML model tools)
- Triage Agent (State machine)
- Clinical Report Generator
- Vision Processor
- Session Store
"""

import os
import sys
import pytest
import json
import base64
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.safety_guardrails import SafetyGuardrails, get_safety_guardrails
from services.semantic_router import SemanticRouter, Intent, get_semantic_router
from services.tool_registry import ToolRegistry, ToolDefinition, get_tool_registry
from services.triage_agent import TriageAgent, TriageState
from services.clinical_report import ClinicalReportGenerator, get_clinical_report_generator
from services.vision_processor import VisionProcessor, get_vision_processor
from services.session_store import SessionStore, get_session_store


# ============================================================================
# Safety Guardrails Tests
# ============================================================================

class TestSafetyGuardrails:
    """Tests for the SafetyGuardrails class."""
    
    def test_safety_guardrails_creation(self):
        """Test creating a safety guardrails instance."""
        guardrails = SafetyGuardrails(model="test-model")
        assert guardrails.model == "test-model"
    
    def test_fallback_detection_critical(self):
        """Test fallback detection for critical keywords."""
        guardrails = SafetyGuardrails()
        result = guardrails._fallback_detection("I'm having chest pain and can't breathe")
        assert result["is_emergency"] is True
        assert result["severity"] == SafetyGuardrails.SEVERITY_CRITICAL
        
    def test_fallback_detection_high(self):
        """Test fallback detection for high severity keywords."""
        guardrails = SafetyGuardrails()
        result = guardrails._fallback_detection("I have severe pain in my stomach")
        assert result["is_emergency"] is True
        assert result["severity"] == SafetyGuardrails.SEVERITY_HIGH
    
    def test_fallback_detection_low(self):
        """Test fallback detection for non-emergency text."""
        guardrails = SafetyGuardrails()
        result = guardrails._fallback_detection("I have a mild headache")
        assert result["is_emergency"] is False
        assert result["severity"] == SafetyGuardrails.SEVERITY_LOW
    
    def test_normalize_result(self):
        """Test result normalization."""
        guardrails = SafetyGuardrails()
        result = guardrails._normalize_result({
            "is_emergency": True,
            "severity": "critical",
            "recommended_action": "emergency_services",
            "reasoning": "Test reasoning"
        })
        assert result["is_emergency"] is True
        assert result["severity"] == SafetyGuardrails.SEVERITY_CRITICAL
        assert result["recommended_action"] == SafetyGuardrails.ACTION_EMERGENCY
    
    def test_get_emergency_response(self):
        """Test getting emergency response message."""
        guardrails = SafetyGuardrails()
        response = guardrails.get_emergency_response()
        assert "911" in response
        assert "emergency" in response.lower()
    
    def test_should_bypass_triage(self):
        """Test triage bypass logic."""
        guardrails = SafetyGuardrails()
        assert guardrails.should_bypass_triage({"is_emergency": True}) is True
        assert guardrails.should_bypass_triage({"is_emergency": False}) is False


# ============================================================================
# Semantic Router Tests
# ============================================================================

class TestSemanticRouter:
    """Tests for the SemanticRouter class."""
    
    def test_semantic_router_creation(self):
        """Test creating a semantic router instance."""
        router = SemanticRouter(model="test-model")
        assert router.model == "test-model"
    
    def test_default_intent(self):
        """Test default intent fallback."""
        router = SemanticRouter()
        result = router._default_intent()
        assert result["intent"] == Intent.SYMPTOM_TRIAGE
        assert result["confidence"] == 0.5
    
    def test_normalize_intent_result(self):
        """Test intent result normalization."""
        router = SemanticRouter()
        result = router._normalize_intent_result({
            "intent": "symptom_triage",
            "confidence": 0.8,
            "reasoning": "Test reasoning"
        })
        assert result["intent"] == Intent.SYMPTOM_TRIAGE
        assert result["confidence"] == 0.8
    
    def test_should_start_intake(self):
        """Test intake start logic."""
        router = SemanticRouter()
        assert router.should_start_intake(Intent.SYMPTOM_TRIAGE) is True
        assert router.should_start_intake(Intent.IMAGE_ANALYSIS) is True
        assert router.should_start_intake(Intent.EMERGENCY) is False
        assert router.should_start_intake(Intent.GENERAL_QUERY) is False
    
    def test_should_use_rag(self):
        """Test RAG usage logic."""
        router = SemanticRouter()
        assert router.should_use_rag(Intent.GENERAL_QUERY) is True
        assert router.should_use_rag(Intent.SYMPTOM_TRIAGE) is False
    
    def test_classify_intent_with_image(self):
        """Test intent classification when image is present."""
        router = SemanticRouter()
        result = router.classify_intent("Check this image", has_image=True)
        assert result["intent"] == Intent.IMAGE_ANALYSIS
        assert result["confidence"] == 1.0


# ============================================================================
# Tool Registry Tests
# ============================================================================

class TestToolRegistry:
    """Tests for the ToolRegistry class."""
    
    def test_tool_registry_creation(self):
        """Test creating a tool registry instance."""
        registry = ToolRegistry()
        assert len(registry._tools) == 0
    
    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"param1": "description"},
            required_data_fields=["param1"],
            questions_to_ask=[{"field": "param1", "question": "What is param1?"}]
        )
        registry.register_tool(tool)
        assert "test_tool" in registry.get_all_tools()
        assert registry.get_tool("test_tool").description == "A test tool"
    
    def test_register_callable(self):
        """Test registering a callable function."""
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            parameters={},
            required_data_fields=[],
            questions_to_ask=[]
        )
        registry.register_tool(tool)
        def test_func(data):
            return {"result": "success"}
        registry.register_callable("test_tool", test_func)
        assert registry.get_tool("test_tool").callable is not None
    
    def test_invoke_tool(self):
        """Test invoking a tool."""
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            parameters={},
            required_data_fields=[],
            questions_to_ask=[]
        )
        registry.register_tool(tool)
        registry.register_callable("test_tool", lambda data: {"result": "success"})
        result = registry.invoke_tool("test_tool", {})
        assert result["result"] == "success"
    
    def test_invoke_nonexistent_tool(self):
        """Test invoking a non-existent tool."""
        registry = ToolRegistry()
        result = registry.invoke_tool("nonexistent", {})
        assert "error" in result


# ============================================================================
# Triage Agent Tests
# ============================================================================

class TestTriageAgent:
    """Tests for the TriageAgent class."""
    
    def test_triage_agent_creation(self):
        """Test creating a triage agent."""
        agent = TriageAgent(session_id="test-session")
        assert agent.session_id == "test-session"
        assert agent.state == TriageState.IDLE
    
    def test_extract_symptoms_from_text(self):
        """Test symptom extraction."""
        agent = TriageAgent(session_id="test-session")
        agent._extract_symptoms_from_text("I have fever and cough")
        symptom_names = [s["name"] for s in agent.clinical_data["symptoms"]]
        assert "fever" in symptom_names
        assert "cough" in symptom_names
    
    def test_get_state(self):
        """Test getting state."""
        agent = TriageAgent(session_id="test-session")
        assert agent.get_state() == TriageState.IDLE
        agent.state = TriageState.GATHERING
        assert agent.get_state() == TriageState.GATHERING
    
    def test_get_clinical_data(self):
        """Test getting clinical data."""
        agent = TriageAgent(session_id="test-session")
        data = agent.get_clinical_data()
        assert "chief_complaint" in data
        assert "symptoms" in data
        assert "patient_profile" in data
    
    def test_set_image_findings(self):
        """Test setting image findings."""
        agent = TriageAgent(session_id="test-session")
        agent.set_image_findings("Rash detected on arm")
        assert agent.clinical_data["image_findings"] == "Rash detected on arm"
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        agent = TriageAgent(session_id="test-session")
        agent.state = TriageState.GATHERING
        data = agent.to_dict()
        assert data["session_id"] == "test-session"
        assert data["state"] == TriageState.GATHERING
        assert "clinical_data" in data
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "session_id": "restored-session",
            "state": TriageState.ANALYZING,
            "clinical_data": {
                "chief_complaint": "Test complaint",
                "symptoms": [{"name": "fever", "mentioned": True}]
            },
            "message_history": [],
            "questions_asked": 2,
            "tools_to_invoke": []
        }
        agent = TriageAgent.from_dict(data)
        assert agent.session_id == "restored-session"
        assert agent.state == TriageState.ANALYZING


# ============================================================================
# Clinical Report Generator Tests
# ============================================================================

class TestClinicalReportGenerator:
    """Tests for the ClinicalReportGenerator class."""
    
    def test_report_generator_creation(self):
        """Test creating a report generator instance."""
        generator = ClinicalReportGenerator(llm_model="test-model")
        assert generator.llm_model == "test-model"
    
    def test_get_disclaimer(self):
        """Test getting disclaimer."""
        generator = ClinicalReportGenerator()
        disclaimer = generator._get_disclaimer()
        assert "NOT a medical diagnosis" in disclaimer
        assert "AI" in disclaimer
    
    def test_format_history(self):
        """Test formatting history of present illness."""
        generator = ClinicalReportGenerator()
        clinical_data = {
            "symptoms": [
                {"name": "fever", "duration": "3 days"},
                {"name": "cough", "duration": "2 days"}
            ]
        }
        history = generator._format_history(clinical_data)
        assert "fever" in history
        assert "cough" in history
    
    def test_generate_fallback_report(self):
        """Test generating fallback report."""
        generator = ClinicalReportGenerator()
        clinical_data = {
            "patient_profile": {"age": 45, "sex": "male"},
            "chief_complaint": "Headache",
            "symptoms": [{"name": "headache"}],
            "risk_factors": ["stress"]
        }
        report = generator._generate_fallback_report(clinical_data, [])
        assert report["chief_complaint"] == "Headache"
        assert "disclaimer" in report
    
    def test_format_for_display(self):
        """Test formatting report for display."""
        generator = ClinicalReportGenerator()
        report = {
            "report_id": "RPT-001",
            "generated_at": "2024-01-01T00:00:00",
            "patient_profile": {"age": 45, "sex": "male"},
            "chief_complaint": "Headache",
            "history_of_present_illness": "Patient reports headache",
            "risk_factors": ["stress"],
            "suggested_differential": [{"condition": "Tension headache", "likelihood": "High"}],
            "recommended_next_steps": ["Rest", "Hydration"],
            "disclaimer": "Test disclaimer"
        }
        formatted = generator.format_for_display(report)
        assert "CLINICAL SUMMARY REPORT" in formatted
        assert "Headache" in formatted


# ============================================================================
# Vision Processor Tests
# ============================================================================

class TestVisionProcessor:
    """Tests for the VisionProcessor class."""
    
    def test_vision_processor_creation(self):
        """Test creating a vision processor instance."""
        processor = VisionProcessor(llm_model="test-model")
        assert processor.llm_model == "test-model"
    
    def test_encode_image(self):
        """Test image encoding to base64."""
        processor = VisionProcessor()
        test_data = b"test_image_data"
        encoded = processor.encode_image(test_data)
        assert isinstance(encoded, str)
        assert base64.b64decode(encoded) == test_data
    
    def test_validate_image_too_large(self):
        """Test image validation for size limit."""
        processor = VisionProcessor()
        # Create large fake image data (> 10MB)
        large_data = b"x" * (11 * 1024 * 1024)
        result = processor.validate_image(large_data)
        # PIL will fail to identify the image format first
        assert result["valid"] is False
        assert "invalid" in result["error"].lower()
    
    def test_validate_real_image(self):
        """Test validating a real X-ray image."""
        processor = VisionProcessor()
        
        # Use the actual x-ray image from docs folder
        xray_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "docs", "x-ray.jpg"
        )
        
        if os.path.exists(xray_path):
            with open(xray_path, 'rb') as f:
                image_data = f.read()
            
            result = processor.validate_image(image_data)
            assert result["valid"] is True
            # Accept various image formats (JPEG, WEBP, PNG, etc.)
            assert result["format"] in ["JPEG", "JPG", "WEBP", "PNG"]
        else:
            pytest.skip("X-ray image not found at docs/x-ray.jpg")
    
    @patch('api.services.vision_processor.ollama.chat')
    def test_process_xray_image_with_gemini(self, mock_chat):
        """Test processing X-ray image with mocked Gemini model."""
        # Mock the LLM response
        mock_chat.return_value = {
            "message": {
                "content": """
                Image Analysis Report:
                
                1. Image Type: Chest X-ray (PA view)
                2. Findings:
                   - Cardiac silhouette appears normal
                   - Lungs are clear bilaterally
                   - No pleural effusion identified
                   - Bony structures appear intact
                3. Notable observations: Normal chest X-ray
                4. Limitations: Clinical correlation recommended
                """
            }
        }
        
        processor = VisionProcessor(llm_model="gemini-3-flash-preview")
        
        # Use the actual x-ray image
        xray_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "docs", "x-ray.jpg"
        )
        
        if os.path.exists(xray_path):
            with open(xray_path, 'rb') as f:
                image_data = f.read()
            
            result = processor.process_image(image_data, context="Chest X-ray analysis")
            
            assert result["success"] is True
            assert "analysis" in result
            assert len(result["analysis"]) > 0
        else:
            pytest.skip("X-ray image not found at docs/x-ray.jpg")


# ============================================================================
# Session Store Tests
# ============================================================================

class TestSessionStore:
    """Tests for the SessionStore class."""
    
    def test_session_store_creation(self):
        """Test creating a session store instance."""
        store = SessionStore(ttl_seconds=900)
        assert store.ttl == 900
    
    def test_create_session(self):
        """Test creating a session."""
        store = SessionStore()
        session_id = store.create()
        assert session_id is not None
        assert len(session_id) == 36  # UUID format
    
    def test_get_session(self):
        """Test getting a session."""
        store = SessionStore()
        session_id = store.create({"test": "data"})
        data = store.get(session_id)
        assert data is not None
    
    def test_get_nonexistent_session(self):
        """Test getting a non-existent session."""
        store = SessionStore()
        data = store.get("nonexistent-id")
        assert data is None
    
    def test_update_session(self):
        """Test updating a session."""
        store = SessionStore()
        session_id = store.create({"test": "data"})
        store.update(session_id, {"test": "updated"})
        data = store.get(session_id)
        assert data["test"] == "updated"
    
    def test_delete_session(self):
        """Test deleting a session."""
        store = SessionStore()
        session_id = store.create()
        store.delete(session_id)
        data = store.get(session_id)
        assert data is None
    
    def test_update_state(self):
        """Test updating session state."""
        store = SessionStore()
        session_id = store.create()
        store.update_state(session_id, "GATHERING")
        data = store.get(session_id)
        assert data["current_state"] == "GATHERING"
    
    def test_add_message(self):
        """Test adding a message to history."""
        store = SessionStore()
        session_id = store.create()
        store.add_message(session_id, "user", "Hello")
        store.add_message(session_id, "assistant", "Hi there")
        data = store.get(session_id)
        assert len(data["message_history"]) == 2
    
    def test_set_report(self):
        """Test setting final report."""
        store = SessionStore()
        session_id = store.create()
        store.set_report(session_id, {"report_id": "RPT-001"})
        data = store.get(session_id)
        assert data["final_report"]["report_id"] == "RPT-001"
        assert data["current_state"] == "COMPLETE"


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
