"""
Integration Tests for Triage API

Test Cases: TC-066 to TC-075
Tests triage API endpoints with mocked services.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask
from unittest.mock import MagicMock, patch


def create_triage_test_app():
    """Create a test Flask app with triage blueprint and mocked services."""
    api_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parent_dir = os.path.dirname(api_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)
    
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


def test_tc066_start_session():
    """TC-066: Start a triage session returns 200 with session_id."""
    print("\n=== TC-066: Start Triage Session ===")
    
    app = create_triage_test_app()
    client = app.test_client()
    
    response = client.post("/api/triage/start", json={})
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.get_json()
    assert data.get("success") is True, "Expected success=True"
    assert "session_id" in data, "Expected session_id in response"
    assert data.get("state") == "INITIAL_INPUT", f"Expected INITIAL_INPUT, got {data.get('state')}"
    
    print(f"[PASS] TC-066: session_id={data.get('session_id')[:16]}..., state={data.get('state')}")
    return True


def test_tc067_start_session_with_id():
    """TC-067: Start session with custom session_id."""
    print("\n=== TC-067: Start Session with Custom ID ===")
    
    app = create_triage_test_app()
    client = app.test_client()
    
    custom_id = "my-custom-session-123"
    response = client.post("/api/triage/start", json={"session_id": custom_id})
    
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("session_id") == custom_id, f"Expected {custom_id}, got {data.get('session_id')}"
    
    print(f"[PASS] TC-067: Custom session_id accepted")
    return True


def test_tc068_send_message_missing_session():
    """TC-068: Send message without session_id returns 400."""
    print("\n=== TC-068: Message Without Session ID ===")
    
    app = create_triage_test_app()
    client = app.test_client()
    
    response = client.post("/api/triage/message", json={"message": "I have a headache"})
    
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    data = response.get_json()
    assert data.get("success") is False
    
    print(f"[PASS] TC-068: Status {response.status_code}")
    return True


def test_tc069_send_message_missing_message():
    """TC-069: Send message without message body returns 400."""
    print("\n=== TC-069: Message Without Body ===")
    
    app = create_triage_test_app()
    client = app.test_client()
    
    response = client.post("/api/triage/message", json={"session_id": "test-123"})
    
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    data = response.get_json()
    assert data.get("success") is False
    
    print(f"[PASS] TC-069: Status {response.status_code}")
    return True


def test_tc070_send_message_valid():
    """TC-070: Send valid message to triage session."""
    print("\n=== TC-070: Send Valid Message ===")
    
    app = create_triage_test_app()
    client = app.test_client()
    
    start_resp = client.post("/api/triage/start", json={})
    session_id = start_resp.get_json()["session_id"]
    
    response = client.post("/api/triage/message", json={
        "session_id": session_id,
        "message": "I have a mild headache"
    })
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.get_json()
    assert data.get("success") is True
    assert "message" in data, "Expected message in response"
    assert data.get("message", "").strip() != "", "Expected non-empty message"
    
    print(f"[PASS] TC-070: Got response, state={data.get('state')}")
    return True


def test_tc071_get_status_existing_session():
    """TC-071: Get status of existing session."""
    print("\n=== TC-071: Get Session Status ===")
    
    app = create_triage_test_app()
    client = app.test_client()
    
    start_resp = client.post("/api/triage/start", json={})
    session_id = start_resp.get_json()["session_id"]
    
    response = client.get(f"/api/triage/status/{session_id}")
    
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("success") is True
    assert data.get("session_id") == session_id
    
    print(f"[PASS] TC-071: Status retrieved for session")
    return True


def test_tc072_get_status_nonexistent_session():
    """TC-072: Get status of nonexistent session returns 404."""
    print("\n=== TC-072: Status of Nonexistent Session ===")
    
    app = create_triage_test_app()
    client = app.test_client()
    
    response = client.get("/api/triage/status/nonexistent-session-id")
    
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"
    data = response.get_json()
    assert data.get("success") is False
    
    print(f"[PASS] TC-072: Status {response.status_code}")
    return True


def test_tc073_get_red_flags():
    """TC-073: Get red flags endpoint returns valid data."""
    print("\n=== TC-073: Get Red Flags ===")
    
    app = create_triage_test_app()
    client = app.test_client()
    
    response = client.get("/api/triage/red-flags")
    
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("success") is True
    assert "red_flags" in data
    
    print(f"[PASS] TC-073: Red flags retrieved")
    return True


def test_tc074_get_urgency_levels():
    """TC-074: Get urgency levels endpoint returns valid data."""
    print("\n=== TC-074: Get Urgency Levels ===")
    
    app = create_triage_test_app()
    client = app.test_client()
    
    response = client.get("/api/triage/urgency-levels")
    
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("success") is True
    assert "urgency_levels" in data
    
    levels = data["urgency_levels"]
    assert "CRITICAL" in levels
    assert "HIGH" in levels
    assert "MODERATE" in levels
    assert "LOW" in levels
    
    print(f"[PASS] TC-074: {len(levels)} urgency levels returned")
    return True


def test_tc075_end_session():
    """TC-075: End session returns report."""
    print("\n=== TC-075: End Triage Session ===")
    
    app = create_triage_test_app()
    client = app.test_client()
    
    start_resp = client.post("/api/triage/start", json={})
    session_id = start_resp.get_json()["session_id"]
    
    response = client.post(f"/api/triage/end/{session_id}")
    
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("success") is True
    
    print(f"[PASS] TC-075: Session ended successfully")
    return True


def run_all_tests():
    """Run all Triage API integration tests."""
    print("=" * 60)
    print("TRIAGE API INTEGRATION TESTS")
    print("=" * 60)
    
    results = []
    tests = [
        ("TC-066", test_tc066_start_session),
        ("TC-067", test_tc067_start_session_with_id),
        ("TC-068", test_tc068_send_message_missing_session),
        ("TC-069", test_tc069_send_message_missing_message),
        ("TC-070", test_tc070_send_message_valid),
        ("TC-071", test_tc071_get_status_existing_session),
        ("TC-072", test_tc072_get_status_nonexistent_session),
        ("TC-073", test_tc073_get_red_flags),
        ("TC-074", test_tc074_get_urgency_levels),
        ("TC-075", test_tc075_end_session),
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
