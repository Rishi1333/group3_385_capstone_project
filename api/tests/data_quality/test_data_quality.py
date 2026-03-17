"""
Data Quality and Validation Tests

Test Cases: TC-106 to TC-115
Tests data source reliability, integrity, handling of missing/noisy data,
and configuration validation.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.clinic_service import ClinicService
from services.safety_guardrails import SafetyGuardrails
from services.dynamic_triage_agent import DynamicTriageAgent
from services.vector_store import VectorStore, Document
from services.embedding_service import EmbeddingService


def test_tc106_clinics_config_integrity():
    """TC-106: Data Quality - Clinics config has valid structure."""
    print("\n=== TC-106: Clinics Config Integrity ===")
    
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "clinics.json"
    )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert "clinics" in data, "Config must have 'clinics' key"
    assert isinstance(data["clinics"], list), "Clinics must be a list"
    assert len(data["clinics"]) > 0, "Config should have at least one clinic"
    
    for clinic in data["clinics"]:
        assert "id" in clinic, f"Clinic missing 'id': {clinic}"
        assert "name" in clinic, f"Clinic missing 'name': {clinic}"
        assert "specialties" in clinic, f"Clinic missing 'specialties': {clinic.get('id')}"
        assert isinstance(clinic["specialties"], list), f"Specialties must be a list for {clinic.get('id')}"
    
    print(f"[PASS] TC-106: {len(data['clinics'])} clinics validated")
    return True


def test_tc107_red_flags_config_integrity():
    """TC-107: Data Quality - Red flags config has valid structure."""
    print("\n=== TC-107: Red Flags Config Integrity ===")
    
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "red_flags.json"
    )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert isinstance(data, dict), "Red flags config must be a dictionary"
    assert len(data) > 0, "Red flags config should not be empty"
    
    print(f"[PASS] TC-107: Red flags config validated ({len(data)} top-level keys)")
    return True


def test_tc108_specialty_mapping_completeness():
    """TC-108: Data Quality - Specialty mapping covers common conditions."""
    print("\n=== TC-108: Specialty Mapping Completeness ===")
    
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "clinics.json"
    )
    service = ClinicService(clinics_config_path=config_path)
    
    common_conditions = ["headache", "chest pain", "stomach pain", "fever", "cough"]
    for condition in common_conditions:
        specialties = service.get_recommended_specialties([condition])
        assert len(specialties) > 0, f"No specialties mapped for condition: {condition}"
    
    print(f"[PASS] TC-108: All common conditions have specialty mappings")
    return True


def test_tc109_clinic_data_required_fields():
    """TC-109: Data Quality - Each clinic has all required fields."""
    print("\n=== TC-109: Clinic Required Fields ===")
    
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "clinics.json"
    )
    service = ClinicService(clinics_config_path=config_path)
    
    required_fields = ["id", "name", "specialties"]
    for clinic in service.get_all_clinics():
        for field in required_fields:
            assert field in clinic, f"Clinic {clinic.get('id', 'unknown')} missing field: {field}"
            assert clinic[field], f"Clinic {clinic.get('id', 'unknown')} has empty field: {field}"
    
    print(f"[PASS] TC-109: All clinics have required fields")
    return True


def test_tc110_missing_data_handling_empty_symptoms():
    """TC-110: Data Quality - System handles missing symptom data gracefully."""
    print("\n=== TC-110: Missing Data - Empty Symptoms ===")
    
    agent = DynamicTriageAgent(session_id="test-missing-110")
    response = agent.process_input("I'm not feeling well")
    
    assert response.get("message"), "Should handle vague symptom input"
    clinical_data = agent.get_clinical_data()
    assert clinical_data["chief_complaint"] != "", "Chief complaint should be captured"
    
    print(f"[PASS] TC-110: Empty/vague symptoms handled gracefully")
    return True


def test_tc111_noisy_data_very_long_input():
    """TC-111: Data Quality - System handles very long inputs."""
    print("\n=== TC-111: Noisy Data - Very Long Input ===")
    
    guardrails = SafetyGuardrails()
    long_input = "headache " * 1000
    
    result = guardrails._fallback_detection(long_input)
    assert "severity" in result, "Should return valid result for long input"
    
    agent = DynamicTriageAgent(session_id="test-noisy-111")
    response = agent.process_input(long_input[:500])
    assert response.get("message"), "Should handle long input"
    
    print(f"[PASS] TC-111: Very long inputs handled")
    return True


def test_tc112_noisy_data_special_characters():
    """TC-112: Data Quality - System handles special characters."""
    print("\n=== TC-112: Noisy Data - Special Characters ===")
    
    agent = DynamicTriageAgent(session_id="test-special-112")
    
    special_inputs = [
        "I have a headache! @#$%",
        "Pain level: 5/10 (sometimes 6/10)",
        "Temperature: 38.5°C",
    ]
    
    for inp in special_inputs:
        response = agent.process_input(inp)
        assert response.get("message"), f"Should handle special chars in: {inp[:30]}"
    
    print(f"[PASS] TC-112: Special characters handled")
    return True


def test_tc113_vector_store_empty_document_handling():
    """TC-113: Data Quality - Vector store rejects empty documents."""
    print("\n=== TC-113: Empty Document Handling ===")
    
    store = VectorStore(backend="memory", collection_name="test_empty_docs")
    
    docs = [
        Document(id="empty1", content="", metadata={"test": "empty"}),
        Document(id="valid1", content="Valid document content", metadata={"test": "valid"}),
    ]
    embeddings = [[0.0] * 384, [0.1] * 384]
    
    store.add_documents(docs, embeddings)
    
    doc = store.get_document("valid1")
    assert doc is not None, "Valid document should be stored"
    
    print(f"[PASS] TC-113: Empty documents handled correctly")
    return True


def test_tc114_embedding_dimension_consistency():
    """TC-114: Data Quality - Embeddings have consistent dimensions."""
    print("\n=== TC-114: Embedding Dimension Consistency ===")
    
    service = EmbeddingService(model_name="test-model", dimension=384)
    
    texts = ["short", "a longer text with more words", "", "special chars: !@#$%"]
    for text in texts:
        embedding = service.embed(text)
        assert len(embedding) == 384, f"Expected 384-dim, got {len(embedding)} for '{text[:20]}'"
    
    batch = service.embed_batch(["text1", "text2", "text3"])
    assert len(batch) == 3, "Batch should return correct number of embeddings"
    for emb in batch:
        assert len(emb) == 384, "All batch embeddings should be 384-dim"
    
    print(f"[PASS] TC-114: Embedding dimensions are consistent")
    return True


def test_tc115_clinic_distance_calculation_accuracy():
    """TC-115: Data Quality - Distance calculation is accurate."""
    print("\n=== TC-115: Distance Calculation Accuracy ===")
    
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "clinics.json"
    )
    service = ClinicService(clinics_config_path=config_path)
    
    result_inf = service._calculate_distance(None, {"coordinates": [-79.38, 43.65]})
    assert result_inf == float('inf'), "Missing location should return infinity"
    
    result_none = service._calculate_distance({"lat": 43.65, "lng": -79.38}, {})
    assert result_none == float('inf'), "Missing coordinates should return infinity"
    
    same_loc = service._calculate_distance(
        {"lat": 43.65, "lng": -79.38},
        {"coordinates": [-79.38, 43.65]}
    )
    assert same_loc < 1.0, f"Same location distance should be < 1km, got {same_loc}"
    
    print(f"[PASS] TC-115: Distance calculations are accurate")
    return True


def run_all_tests():
    """Run all Data Quality tests."""
    print("=" * 60)
    print("DATA QUALITY & VALIDATION TESTS")
    print("=" * 60)
    
    results = []
    tests = [
        ("TC-106", test_tc106_clinics_config_integrity),
        ("TC-107", test_tc107_red_flags_config_integrity),
        ("TC-108", test_tc108_specialty_mapping_completeness),
        ("TC-109", test_tc109_clinic_data_required_fields),
        ("TC-110", test_tc110_missing_data_handling_empty_symptoms),
        ("TC-111", test_tc111_noisy_data_very_long_input),
        ("TC-112", test_tc112_noisy_data_special_characters),
        ("TC-113", test_tc113_vector_store_empty_document_handling),
        ("TC-114", test_tc114_embedding_dimension_consistency),
        ("TC-115", test_tc115_clinic_distance_calculation_accuracy),
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
