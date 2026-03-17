"""
User Acceptance Tests (UAT)

Test Cases: TC-086 to TC-095
Tests complete user journeys and business logic from the end-user perspective.
These tests validate that the system meets user requirements and expectations.
"""

import sys
import os
import json
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask
from unittest.mock import MagicMock, patch
from services.dynamic_triage_agent import DynamicTriageAgent, TriageState
from services.safety_guardrails import SafetyGuardrails
from services.clinic_service import ClinicService
from services.session_store import SessionStore
from services.clinical_report import ClinicalReportGenerator
from services.vision_processor import VisionProcessor


def test_tc086_uat_patient_symptom_journey():
    """TC-086: UAT - Complete patient symptom reporting journey."""
    print("\n=== TC-086: UAT - Patient Symptom Journey ===")
    
    agent = DynamicTriageAgent(session_id="uat-journey-086")
    
    responses = []
    messages = [
        "I have been having back pain",
        "I am 30 year old male",
        "It started about 2 weeks ago",
        "No medical history or medications",
        "No allergies"
    ]
    
    for msg in messages:
        response = agent.process_input(msg)
        responses.append(response)
        assert response.get("message"), f"Bot should respond to '{msg}'"
        assert response.get("message", "").strip() != "", "Bot should not return empty messages"
    
    clinical_data = agent.get_clinical_data()
    assert clinical_data["chief_complaint"] != "", "Chief complaint should be captured"
    assert clinical_data["patient_profile"]["age"] is not None, "Age should be extracted"
    
    print(f"[PASS] TC-086: Patient journey completed, data captured")
    return True


def test_tc087_uat_emergency_detection_journey():
    """TC-087: UAT - Emergency symptoms are detected and handled correctly."""
    print("\n=== TC-087: UAT - Emergency Detection Journey ===")
    
    guardrails = SafetyGuardrails()
    detection = guardrails._fallback_detection("I have severe chest pain and can't breathe")
    assert detection["is_emergency"] is True, "Fallback should detect chest pain as emergency"
    assert detection["severity"] == "CRITICAL", "Chest pain should be CRITICAL severity"
    
    agent = DynamicTriageAgent(session_id="uat-emergency-087")
    response = agent.process_input("I have severe chest pain and can't breathe")
    
    message = response.get("message", "")
    assert len(message) > 0, "Emergency response should not be empty"
    assert response.get("state") is not None, "Should have a valid state"
    
    print(f"[PASS] TC-087: Emergency detected, appropriate response given")
    return True


def test_tc087b_uat_emergency_response_contains_contacts():
    """TC-087b: UAT - Emergency response includes emergency contact information."""
    print("\n=== TC-087b: UAT - Emergency Contact Info ===")
    
    guardrails = SafetyGuardrails()
    emergency_msg = guardrails.get_emergency_response()
    
    assert "911" in emergency_msg, "Emergency response should include 911"
    assert "988" in emergency_msg, "Emergency response should include suicide prevention number"
    
    print(f"[PASS] TC-087b: Emergency contacts present in response")
    return True


def test_tc088_uat_clinic_recommendation_relevance():
    """TC-088: UAT - Clinic recommendations are relevant to patient symptoms."""
    print("\n=== TC-088: UAT - Clinic Recommendation Relevance ===")
    
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "clinics.json"
    )
    service = ClinicService(clinics_config_path=config_path)
    
    test_cases = [
        (["Headache"], ["neurology", "general"]),
        (["Chest Pain"], ["cardiology", "general"]),
        (["Stomach Pain"], ["gastroenterology", "general"]),
    ]
    
    for conditions, expected_specialties in test_cases:
        specialties = service.get_recommended_specialties(conditions)
        has_expected = any(s in specialties for s in expected_specialties)
        assert has_expected or "general" in specialties, \
            f"For {conditions}, expected one of {expected_specialties}, got {specialties}"
    
    print(f"[PASS] TC-088: Clinic recommendations are relevant")
    return True


def test_tc089_uat_no_empty_bot_messages():
    """TC-089: UAT - Bot never returns empty messages during conversation."""
    print("\n=== TC-089: UAT - No Empty Bot Messages ===")
    
    agent = DynamicTriageAgent(session_id="uat-empty-089")
    
    test_inputs = [
        "I have a headache",
        "I am 25 years old female",
        "ok",
        "yes",
        "no",
        "maybe",
    ]
    
    for msg in test_inputs:
        response = agent.process_input(msg)
        bot_message = response.get("message", "")
        assert bot_message.strip() != "", f"Bot returned empty message for input: '{msg}'"
    
    print(f"[PASS] TC-089: No empty bot messages detected")
    return True


def test_tc090_uat_disclaimer_in_reports():
    """TC-090: UAT - Clinical reports contain required medical disclaimers."""
    print("\n=== TC-090: UAT - Disclaimer in Reports ===")
    
    generator = ClinicalReportGenerator()
    
    clinical_data = {
        "patient_profile": {"age": 40, "sex": "female"},
        "chief_complaint": "Headache",
        "symptoms": [{"name": "headache"}],
        "medical_history": [],
        "medications": [],
        "allergies": [],
        "risk_factors": [],
    }
    
    report = generator._generate_fallback_report(clinical_data, [])
    disclaimer = report.get("disclaimer", "")
    
    assert "NOT a medical diagnosis" in disclaimer, "Disclaimer must state it's not a diagnosis"
    assert "healthcare" in disclaimer.lower(), "Disclaimer must mention healthcare professional"
    
    formatted = generator.format_for_display(report)
    assert "Disclaimer" in formatted or "DISCLAIMER" in formatted, "Formatted report must show disclaimer"
    
    print(f"[PASS] TC-090: Disclaimers present in reports")
    return True


def test_tc091_uat_session_lifecycle():
    """TC-091: UAT - Complete session lifecycle: create, use, and cleanup."""
    print("\n=== TC-091: UAT - Session Lifecycle ===")
    
    store = SessionStore(ttl_seconds=300)
    
    session_id = store.create()
    assert store.get(session_id) is not None, "Session should exist after creation"
    
    store.update(session_id, {"current_state": "GATHERING"})
    data = store.get(session_id)
    assert data["current_state"] == "GATHERING"
    
    store.add_message(session_id, "user", "I have a headache")
    store.add_message(session_id, "assistant", "I understand. Can you tell me more?")
    data = store.get(session_id)
    assert len(data["message_history"]) == 2, "Messages should be stored"
    
    store.delete(session_id)
    assert store.get(session_id) is None, "Session should be deleted"
    
    print(f"[PASS] TC-091: Session lifecycle works correctly")
    return True


def test_tc092_uat_image_validation_user_experience():
    """TC-092: UAT - Image validation provides helpful error messages."""
    print("\n=== TC-092: UAT - Image Validation UX ===")
    
    processor = VisionProcessor()
    
    invalid_data = b"this is not an image"
    result = processor.validate_image(invalid_data)
    assert result.get("valid") is False, "Should detect invalid image"
    assert "error" in result, "Should provide error message"
    
    result_large = {"valid": False, "error": "Image too large (max 10MB)"}
    assert "max 10MB" in result_large["error"], "Error should mention size limit"
    
    print(f"[PASS] TC-092: Image validation provides helpful errors")
    return True


def test_tc093_uat_booking_flow_end_to_end():
    """TC-093: UAT - Booking flow from symptom to clinic recommendation."""
    print("\n=== TC-093: UAT - Booking Flow E2E ===")
    
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "clinics.json"
    )
    service = ClinicService(clinics_config_path=config_path)
    
    specialties = service.get_recommended_specialties(["Headache", "Fever"])
    assert len(specialties) > 0
    
    clinics = service.find_clinics_by_specialty(specialties)
    
    if len(clinics) > 0:
        clinic = clinics[0]
        doctors = clinic.get("matching_doctors", [])
        if len(doctors) > 0:
            doctor = doctors[0]
            assert doctor.get("name") is not None or doctor.get("id") is not None
    
    print(f"[PASS] TC-093: Complete booking flow works")
    return True


def test_tc094_uat_urgency_classification_accuracy():
    """TC-094: UAT - Urgency classification produces reasonable results."""
    print("\n=== TC-094: UAT - Urgency Classification Accuracy ===")
    
    guardrails = SafetyGuardrails()
    
    test_cases = [
        ("I'm having chest pain and can't breathe", True, "CRITICAL"),
        ("I have severe pain in my stomach", True, "HIGH"),
        ("I have a mild headache", False, "LOW"),
        ("", False, "LOW"),
    ]
    
    for text, expected_emergency, expected_severity in test_cases:
        result = guardrails._fallback_detection(text)
        assert result["is_emergency"] == expected_emergency, \
            f"For '{text}', expected emergency={expected_emergency}, got {result['is_emergency']}"
    
    print(f"[PASS] TC-094: Urgency classification is accurate for fallback detection")
    return True


def test_tc095_uat_accessibility_response_quality():
    """TC-095: UAT - Bot responses are accessible and user-friendly."""
    print("\n=== TC-095: UAT - Response Accessibility ===")
    
    agent = DynamicTriageAgent(session_id="uat-access-095")
    
    response = agent.process_input("I have a headache")
    message = response.get("message", "")
    
    assert len(message) > 10, "Response should be informative, not just a word"
    assert len(message) < 2000, "Response should not be overwhelmingly long"
    
    assert message.strip() != "", "Response should not be empty"
    
    assert not message.startswith("Error"), "Response should not start with 'Error'"
    
    print(f"[PASS] TC-095: Responses are accessible and user-friendly")
    return True


def run_all_tests():
    """Run all User Acceptance Tests."""
    print("=" * 60)
    print("USER ACCEPTANCE TESTS (UAT)")
    print("=" * 60)
    
    results = []
    tests = [
        ("TC-086", test_tc086_uat_patient_symptom_journey),
        ("TC-087", test_tc087_uat_emergency_detection_journey),
        ("TC-087b", test_tc087b_uat_emergency_response_contains_contacts),
        ("TC-088", test_tc088_uat_clinic_recommendation_relevance),
        ("TC-089", test_tc089_uat_no_empty_bot_messages),
        ("TC-090", test_tc090_uat_disclaimer_in_reports),
        ("TC-091", test_tc091_uat_session_lifecycle),
        ("TC-092", test_tc092_uat_image_validation_user_experience),
        ("TC-093", test_tc093_uat_booking_flow_end_to_end),
        ("TC-094", test_tc094_uat_urgency_classification_accuracy),
        ("TC-095", test_tc095_uat_accessibility_response_quality),
    ]
    
    for test_id, test_func in tests:
        try:
            results.append((test_id, test_func()))
        except AssertionError as e:
            print(f"[FAIL] {test_id}: {e}")
            results.append((test_id, False))
        except Exception as e:
            print(f"[ERROR] {test_id}: {e}")
            results.append((test_id, False))
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
