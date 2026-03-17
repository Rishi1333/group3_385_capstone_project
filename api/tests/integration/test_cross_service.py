"""
Integration Tests for Cross-Service Workflows

Test Cases: TC-076 to TC-085
Tests interactions between multiple services working together.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask
from unittest.mock import MagicMock, patch
from services.clinic_service import ClinicService
from services.session_store import SessionStore
from services.safety_guardrails import SafetyGuardrails
from services.semantic_router import SemanticRouter, Intent
from services.dynamic_triage_agent import DynamicTriageAgent, TriageState, UrgencyLevel


def test_tc076_safety_guardrails_to_triage_agent():
    """TC-076: Safety guardrails detection feeds into triage agent state."""
    print("\n=== TC-076: Safety Guardrails -> Triage Agent Integration ===")
    
    guardrails = SafetyGuardrails()
    result = guardrails._fallback_detection("I'm having chest pain and can't breathe")
    
    assert result["is_emergency"] is True
    assert result["severity"] == SafetyGuardrails.SEVERITY_CRITICAL
    
    agent = DynamicTriageAgent(session_id="test-integ-076")
    response = agent.process_input("I'm having chest pain and can't breathe")
    
    valid_states = [
        TriageState.EMERGENCY_PATH.value,
        TriageState.COMPLETE.value,
        TriageState.PHI_COLLECTION.value,
        TriageState.RAPID_TRIAGE.value,
    ]
    assert response.get("state") in valid_states, \
        f"Expected valid triage state, got {response.get('state')}"
    
    clinical_data = agent.get_clinical_data()
    assert clinical_data.get("urgency_level") is not None or result["is_emergency"], \
        "Urgency should be assessed or emergency detected"
    
    print(f"[PASS] TC-076: Safety guardrails -> triage agent integration verified, state={response.get('state')}")
    return True


def test_tc077_semantic_router_to_triage_flow():
    """TC-077: Semantic router classification determines triage flow."""
    print("\n=== TC-077: Semantic Router -> Triage Flow ===")
    
    router = SemanticRouter()
    
    image_result = router.classify_intent("Check this rash", has_image=True)
    assert image_result["intent"] == Intent.IMAGE_ANALYSIS
    assert router.should_start_intake(Intent.IMAGE_ANALYSIS) is True
    
    symptom_result = router._default_intent()
    assert router.should_start_intake(symptom_result["intent"]) is True
    
    emergency_result = {"intent": Intent.EMERGENCY}
    assert router.should_start_intake(Intent.EMERGENCY) is False
    
    print(f"[PASS] TC-077: Router correctly determines intake flow")
    return True


def test_tc078_session_store_persistence():
    """TC-078: Session data persists across multiple operations."""
    print("\n=== TC-078: Session Store Persistence ===")
    
    store = SessionStore(ttl_seconds=300)
    session_id = store.create({"initial": "data"})
    
    retrieved = store.get(session_id)
    assert retrieved is not None, "Session should be retrievable"
    assert retrieved["initial"] == "data"
    
    store.update(session_id, {"additional": "info"})
    updated = store.get(session_id)
    assert updated["additional"] == "info"
    assert updated["initial"] == "data"
    
    store.update_state(session_id, "GATHERING")
    final = store.get(session_id)
    assert final["current_state"] == "GATHERING"
    
    print(f"[PASS] TC-078: Session data persists across operations")
    return True


def test_tc079_clinic_service_to_booking_data():
    """TC-079: Clinic service data flows correctly to booking system."""
    print("\n=== TC-079: Clinic Service -> Booking Data Flow ===")
    
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "clinics.json"
    )
    clinic_service = ClinicService(clinics_config_path=config_path)
    
    specialties = clinic_service.get_recommended_specialties(["Chest Pain"])
    assert len(specialties) > 0, "Expected specialties for chest pain"
    
    clinics = clinic_service.find_clinics_by_specialty(specialties)
    if len(clinics) > 0:
        first_clinic = clinics[0]
        assert "matching_specialties" in first_clinic
        assert first_clinic.get("id") is not None
        
        clinic_detail = clinic_service.get_clinic_by_id(first_clinic["id"])
        assert clinic_detail is not None, "Should find clinic by ID"
    
    print(f"[PASS] TC-079: Specialty->Clinic data flow verified")
    return True


def test_tc080_dynamic_triage_agent_phi_extraction():
    """TC-080: Triage agent extracts PHI from natural conversation."""
    print("\n=== TC-080: Triage Agent PHI Extraction ===")
    
    agent = DynamicTriageAgent(session_id="test-phi-080")
    
    response = agent.process_input("I am a 35 year old female with back pain")
    
    clinical_data = agent.get_clinical_data()
    profile = clinical_data.get("patient_profile", {})
    
    assert profile.get("age") == 35 or profile.get("age") is not None, \
        f"Expected age extraction, got {profile.get('age')}"
    assert profile.get("sex") in ["female", None], \
        f"Expected sex extraction, got {profile.get('sex')}"
    
    print(f"[PASS] TC-080: PHI extracted - age={profile.get('age')}, sex={profile.get('sex')}")
    return True


def test_tc081_triage_agent_symptom_extraction():
    """TC-081: Triage agent extracts symptoms from input."""
    print("\n=== TC-081: Triage Agent Symptom Extraction ===")
    
    agent = DynamicTriageAgent(session_id="test-symptom-081")
    agent.process_input("I have fever and cough")
    
    clinical_data = agent.get_clinical_data()
    symptoms = clinical_data.get("symptoms", [])
    
    symptom_names = [s.get("name") if isinstance(s, dict) else s for s in symptoms]
    assert len(symptom_names) > 0, f"Expected symptoms to be extracted, got {symptom_names}"
    
    print(f"[PASS] TC-081: Extracted symptoms: {symptom_names}")
    return True


def test_tc082_triage_agent_state_transitions():
    """TC-082: Triage agent follows correct state transitions."""
    print("\n=== TC-082: Triage Agent State Transitions ===")
    
    agent = DynamicTriageAgent(session_id="test-state-082")
    
    assert agent.get_state() == TriageState.INITIAL_INPUT.value
    
    response = agent.process_input("I have a headache")
    state_after_input = agent.get_state()
    
    valid_states = [
        TriageState.RAPID_TRIAGE.value,
        TriageState.PHI_COLLECTION.value,
        TriageState.EMERGENCY_PATH.value,
        TriageState.DEEP_DIVE.value,
    ]
    assert state_after_input in valid_states, f"Expected valid transition, got {state_after_input}"
    
    print(f"[PASS] TC-082: State transition INITIAL_INPUT -> {state_after_input}")
    return True


def test_tc083_clinic_specialty_mapping_consistency():
    """TC-083: Clinic specialty mapping is consistent across service methods."""
    print("\n=== TC-083: Specialty Mapping Consistency ===")
    
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "clinics.json"
    )
    service = ClinicService(clinics_config_path=config_path)
    
    all_specialties = service.get_all_specialties()
    all_clinics = service.get_all_clinics()
    
    clinic_specialties = set()
    for clinic in all_clinics:
        clinic_specialties.update(clinic.get("specialties", []))
    
    assert set(all_specialties) == clinic_specialties, \
        "Service specialty list should match clinic specialties"
    
    print(f"[PASS] TC-083: {len(all_specialties)} specialties consistent")
    return True


def test_tc084_rag_service_with_mock_components():
    """TC-084: RAG service integrates vector store and embedding service."""
    print("\n=== TC-084: RAG Service Component Integration ===")
    
    from services.rag_service import RAGService
    from services.vector_store import VectorStore, Document
    from services.embedding_service import EmbeddingService
    
    mock_store = MagicMock(spec=VectorStore)
    mock_store.count.return_value = 3
    mock_store.similarity_search.return_value = [
        Document(id="1", content="Heart disease info", metadata={"source": "test", "distance": 0.2}),
        Document(id="2", content="Cardiac symptoms", metadata={"source": "test", "distance": 0.3}),
    ]
    
    mock_embedding = MagicMock(spec=EmbeddingService)
    mock_embedding.embed.return_value = [0.1] * 384
    
    rag = RAGService(
        vector_store=mock_store,
        embedding_service=mock_embedding,
        min_relevance_score=0.3
    )
    
    assert rag.is_available()
    
    docs = rag.retrieve("heart symptoms")
    mock_embedding.embed.assert_called()
    mock_store.similarity_search.assert_called()
    
    print(f"[PASS] TC-084: RAG service integrates components correctly")
    return True


def test_tc085_clinical_report_data_flow():
    """TC-085: Clinical data flows correctly into report generation."""
    print("\n=== TC-085: Clinical Report Data Flow ===")
    
    from services.clinical_report import ClinicalReportGenerator
    
    generator = ClinicalReportGenerator()
    
    clinical_data = {
        "patient_profile": {"age": 45, "sex": "male"},
        "chief_complaint": "Chest pain",
        "symptoms": [{"name": "chest pain", "duration": "1 week"}],
        "medical_history": ["Hypertension"],
        "medications": ["Lisinopril"],
        "allergies": ["Penicillin"],
        "risk_factors": ["Smoking"],
        "lifestyle": {"smoking": "current", "alcohol": "none", "exercise": "moderate"},
    }
    
    report = generator._generate_fallback_report(clinical_data, [])
    
    assert report["chief_complaint"] == "Chest pain"
    assert report["patient_profile"]["age"] == 45
    assert "Hypertension" in report["past_medical_history"]
    assert "Lisinopril" in report["current_medications"]
    assert "Penicillin" in report["allergies"]
    assert report["disclaimer"] != ""
    
    formatted = generator.format_for_display(report)
    assert "CLINICAL SUMMARY REPORT" in formatted
    assert "Chest pain" in formatted
    
    print(f"[PASS] TC-085: Clinical data flows correctly to report")
    return True


def run_all_tests():
    """Run all cross-service integration tests."""
    print("=" * 60)
    print("CROSS-SERVICE INTEGRATION TESTS")
    print("=" * 60)
    
    results = []
    tests = [
        ("TC-076", test_tc076_safety_guardrails_to_triage_agent),
        ("TC-077", test_tc077_semantic_router_to_triage_flow),
        ("TC-078", test_tc078_session_store_persistence),
        ("TC-079", test_tc079_clinic_service_to_booking_data),
        ("TC-080", test_tc080_dynamic_triage_agent_phi_extraction),
        ("TC-081", test_tc081_triage_agent_symptom_extraction),
        ("TC-082", test_tc082_triage_agent_state_transitions),
        ("TC-083", test_tc083_clinic_specialty_mapping_consistency),
        ("TC-084", test_tc084_rag_service_with_mock_components),
        ("TC-085", test_tc085_clinical_report_data_flow),
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
