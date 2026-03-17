"""
Integration Tests for Booking API

Test Cases: TC-017, TC-018, TC-019, TC-020
Tests booking API endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask
from unittest.mock import MagicMock
from routes.booking import create_booking_blueprint
from services.clinic_service import get_clinic_service


def create_test_app():
    """Create a test Flask app."""
    app = Flask(__name__)
    
    # Mock database
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=MagicMock())
    
    clinic_service = get_clinic_service()
    booking_bp = create_booking_blueprint(clinic_service, mock_db)
    app.register_blueprint(booking_bp)
    
    return app


def test_tc017_recommend_endpoint():
    """TC-017: Test /api/booking/recommend endpoint"""
    print("\n=== TC-017: Booking Recommend Endpoint ===")
    
    app = create_test_app()
    client = app.test_client()
    
    response = client.post(
        "/api/booking/recommend",
        json={"conditions": ["Headache"]}
    )
    
    # Assertions
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.get_json()
    assert "specialties" in data, "Expected 'specialties' in response"
    assert "clinics" in data, "Expected 'clinics' in response"
    
    print(f"Input: {{\"conditions\": [\"Headache\"]}}")
    print(f"Status: {response.status_code}")
    print(f"Response keys: {list(data.keys())}")
    print("[PASS] TC-017 passed")
    return True


def test_tc018_recommend_missing_conditions():
    """TC-018: Test recommend endpoint with missing conditions"""
    print("\n=== TC-018: Recommend Missing Conditions ===")
    
    app = create_test_app()
    client = app.test_client()
    
    response = client.post(
        "/api/booking/recommend",
        json={}
    )
    
    # Assertions
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    data = response.get_json()
    assert "error" in data, "Expected 'error' in response"
    
    print(f"Input: {{}}")
    print(f"Status: {response.status_code}")
    print(f"Error: {data.get('error')}")
    print("[PASS] TC-018 passed")
    return True


def test_tc019_clinics_endpoint():
    """TC-019: Test /api/booking/clinics endpoint"""
    print("\n=== TC-019: Clinics Endpoint ===")
    
    app = create_test_app()
    client = app.test_client()
    
    response = client.get("/api/booking/clinics")
    
    # Assertions
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.get_json()
    assert "clinics" in data, "Expected 'clinics' in response"
    assert len(data["clinics"]) > 0, "Expected at least one clinic"
    
    print(f"Status: {response.status_code}")
    print(f"Clinics count: {len(data['clinics'])}")
    print("[PASS] TC-019 passed")
    return True


def test_tc020_clinic_not_found():
    """TC-020: Test clinic by ID with nonexistent clinic"""
    print("\n=== TC-020: Clinic Not Found ===")
    
    app = create_test_app()
    client = app.test_client()
    
    response = client.get("/api/booking/clinics/nonexistent")
    
    # Assertions
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"
    
    print(f"Input: \"/api/booking/clinics/nonexistent\"")
    print(f"Status: {response.status_code}")
    print("[PASS] TC-020 passed")
    return True


def run_all_tests():
    """Run all Booking API tests."""
    print("=" * 60)
    print("BOOKING API INTEGRATION TESTS")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("TC-017", test_tc017_recommend_endpoint),
        ("TC-018", test_tc018_recommend_missing_conditions),
        ("TC-019", test_tc019_clinics_endpoint),
        ("TC-020", test_tc020_clinic_not_found),
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
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_id, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {test_id}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
