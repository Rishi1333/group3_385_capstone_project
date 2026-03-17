"""
Integration Tests for Auth API

Test Cases: TC-056 to TC-065
Tests authentication API endpoints with mocked MongoDB.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask
from unittest.mock import MagicMock, patch
from bson import ObjectId

import mongo_db as mongo_db_module


def create_auth_test_app():
    """Create a test Flask app with auth blueprint and mocked DB."""
    app = Flask(__name__)
    app.config["JWT_SECRET_KEY"] = "test_secret_key"
    
    from flask_jwt_extended import JWTManager
    JWTManager(app)
    
    mock_db = MagicMock()
    mock_patients = MagicMock()
    mock_doctors = MagicMock()
    mock_admin = MagicMock()
    
    mock_db.__getitem__ = MagicMock(side_effect=lambda key: {
        "patients": mock_patients,
        "doctors": mock_doctors,
        "administrator": mock_admin,
    }.get(key, MagicMock()))
    
    mongo_db_module._db = mock_db
    
    from routes.auth import auth_bp
    app.register_blueprint(auth_bp)
    
    return app, mock_db


def test_tc056_register_missing_fields():
    """TC-056: Register with missing email and password returns 400."""
    print("\n=== TC-056: Register Missing Fields ===")
    
    app, mock_db = create_auth_test_app()
    mock_db["patients"].find_one.return_value = None
    
    client = app.test_client()
    response = client.post("/auth/register", json={})
    
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    data = response.get_json()
    assert "error" in data, "Expected error in response"
    
    print(f"[PASS] TC-056: Status {response.status_code}, error: {data.get('error')}")
    return True


def test_tc057_register_valid_patient():
    """TC-057: Register a valid patient returns 201."""
    print("\n=== TC-057: Register Valid Patient ===")
    
    app, mock_db = create_auth_test_app()
    mock_db["patients"].find_one.return_value = None
    mock_db["administrator"].find_one.return_value = None
    mock_db["doctors"].find_one.return_value = None
    mock_db["patients"].insert_one.return_value = MagicMock(inserted_id=ObjectId())
    
    client = app.test_client()
    response = client.post("/auth/register", json={
        "email": "test_patient@example.com",
        "password": "SecurePass123",
        "full_name": "Test Patient"
    })
    
    assert response.status_code == 201, f"Expected 201, got {response.status_code}"
    data = response.get_json()
    assert "token" in data, "Expected token in response"
    assert data.get("role") == "patients", f"Expected role=patients, got {data.get('role')}"
    
    print(f"[PASS] TC-057: Status {response.status_code}, role={data.get('role')}")
    return True


def test_tc058_register_duplicate_email():
    """TC-058: Register with duplicate email returns 409."""
    print("\n=== TC-058: Register Duplicate Email ===")
    
    app, mock_db = create_auth_test_app()
    mock_db["patients"].find_one.return_value = {
        "_id": ObjectId(),
        "email": "existing@example.com",
        "password": b"$2b$12$hashed",
        "role": "patients"
    }
    mock_db["administrator"].find_one.return_value = None
    mock_db["doctors"].find_one.return_value = None
    
    client = app.test_client()
    response = client.post("/auth/register", json={
        "email": "existing@example.com",
        "password": "SecurePass123",
        "full_name": "Duplicate User"
    })
    
    assert response.status_code == 409, f"Expected 409, got {response.status_code}"
    data = response.get_json()
    assert "error" in data, "Expected error in response"
    
    print(f"[PASS] TC-058: Status {response.status_code}, error: {data.get('error')}")
    return True


def test_tc059_login_missing_fields():
    """TC-059: Login with missing fields returns 400."""
    print("\n=== TC-059: Login Missing Fields ===")
    
    app, mock_db = create_auth_test_app()
    client = app.test_client()
    
    response = client.post("/auth/login", json={})
    
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    data = response.get_json()
    assert "error" in data
    
    print(f"[PASS] TC-059: Status {response.status_code}")
    return True


def test_tc060_login_nonexistent_user():
    """TC-060: Login with nonexistent user returns 404."""
    print("\n=== TC-060: Login Nonexistent User ===")
    
    app, mock_db = create_auth_test_app()
    mock_db["patients"].find_one.return_value = None
    mock_db["doctors"].find_one.return_value = None
    mock_db["administrator"].find_one.return_value = None
    
    client = app.test_client()
    response = client.post("/auth/login", json={
        "email": "nobody@example.com",
        "password": "SomePassword123"
    })
    
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"
    data = response.get_json()
    assert "error" in data
    
    print(f"[PASS] TC-060: Status {response.status_code}")
    return True


def test_tc061_login_wrong_password():
    """TC-061: Login with wrong password returns 401."""
    print("\n=== TC-061: Login Wrong Password ===")
    
    import bcrypt
    app, mock_db = create_auth_test_app()
    
    hashed = bcrypt.hashpw("CorrectPass123".encode("utf-8"), bcrypt.gensalt())
    mock_db["patients"].find_one.return_value = {
        "_id": ObjectId(),
        "email": "user@example.com",
        "password": hashed,
        "full_name": "Test User",
        "role": "patients",
        "gender": None
    }
    mock_db["doctors"].find_one.return_value = None
    mock_db["administrator"].find_one.return_value = None
    
    client = app.test_client()
    response = client.post("/auth/login", json={
        "email": "user@example.com",
        "password": "WrongPassword"
    })
    
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"
    data = response.get_json()
    assert "error" in data
    
    print(f"[PASS] TC-061: Status {response.status_code}")
    return True


def test_tc062_login_correct_credentials():
    """TC-062: Login with correct credentials returns 200 and JWT token."""
    print("\n=== TC-062: Login Correct Credentials ===")
    
    import bcrypt
    app, mock_db = create_auth_test_app()
    
    hashed = bcrypt.hashpw("CorrectPass123".encode("utf-8"), bcrypt.gensalt())
    mock_db["patients"].find_one.return_value = {
        "_id": ObjectId(),
        "email": "user@example.com",
        "password": hashed,
        "full_name": "Test User",
        "role": "patients",
        "gender": None
    }
    mock_db["doctors"].find_one.return_value = None
    mock_db["administrator"].find_one.return_value = None
    
    client = app.test_client()
    response = client.post("/auth/login", json={
        "email": "user@example.com",
        "password": "CorrectPass123"
    })
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.get_json()
    assert "token" in data, "Expected JWT token in response"
    assert data.get("role") == "patients"
    
    print(f"[PASS] TC-062: Status {response.status_code}, token present")
    return True


def test_tc063_password_hashing():
    """TC-063: Verify passwords are hashed, not stored in plaintext."""
    print("\n=== TC-063: Password Hashing Verification ===")
    
    from routes.auth import _hash_password, _check_password
    
    plain = "MySecurePassword123"
    hashed = _hash_password(plain)
    
    assert hashed != plain.encode("utf-8"), "Password should not be stored as plaintext"
    assert _check_password(plain, hashed), "Hashed password should verify correctly"
    assert not _check_password("WrongPassword", hashed), "Wrong password should not verify"
    
    print(f"[PASS] TC-063: Password hashing verified")
    return True


def test_tc064_auth_blueprint_routes():
    """TC-064: Verify all auth routes are registered."""
    print("\n=== TC-064: Auth Blueprint Routes ===")
    
    app, _ = create_auth_test_app()
    rules = [rule.rule for rule in app.url_map.iter_rules()]
    
    auth_routes = [r for r in rules if r.startswith("/auth/")]
    assert len(auth_routes) > 0, "Expected auth routes to be registered"
    
    expected = ["/auth/register", "/auth/login", "/auth/profile"]
    for route in expected:
        assert route in rules, f"Expected route {route} not found"
    
    print(f"[PASS] TC-064: {len(auth_routes)} auth routes registered")
    return True


def test_tc065_jwt_token_structure():
    """TC-065: Verify JWT token contains required claims."""
    print("\n=== TC-065: JWT Token Structure ===")
    
    import bcrypt
    import base64
    
    app, mock_db = create_auth_test_app()
    
    hashed = bcrypt.hashpw("Pass123".encode("utf-8"), bcrypt.gensalt())
    user_id = ObjectId()
    mock_db["patients"].find_one.return_value = {
        "_id": user_id,
        "email": "jwt_test@example.com",
        "password": hashed,
        "full_name": "JWT Test",
        "role": "patients",
        "gender": "male"
    }
    mock_db["doctors"].find_one.return_value = None
    mock_db["administrator"].find_one.return_value = None
    
    client = app.test_client()
    response = client.post("/auth/login", json={
        "email": "jwt_test@example.com",
        "password": "Pass123"
    })
    
    assert response.status_code == 200
    data = response.get_json()
    token = data.get("token", "")
    
    parts = token.split(".")
    assert len(parts) == 3, f"Expected JWT with 3 parts, got {len(parts)}"
    
    payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
    payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    
    assert "sub" in payload, "Expected 'sub' claim in JWT"
    assert "role" in payload, "Expected 'role' claim in JWT"
    
    print(f"[PASS] TC-065: JWT contains sub={payload.get('sub')}, role={payload.get('role')}")
    return True


def run_all_tests():
    """Run all Auth API integration tests."""
    print("=" * 60)
    print("AUTH API INTEGRATION TESTS")
    print("=" * 60)
    
    results = []
    tests = [
        ("TC-056", test_tc056_register_missing_fields),
        ("TC-057", test_tc057_register_valid_patient),
        ("TC-058", test_tc058_register_duplicate_email),
        ("TC-059", test_tc059_login_missing_fields),
        ("TC-060", test_tc060_login_nonexistent_user),
        ("TC-061", test_tc061_login_wrong_password),
        ("TC-062", test_tc062_login_correct_credentials),
        ("TC-063", test_tc063_password_hashing),
        ("TC-064", test_tc064_auth_blueprint_routes),
        ("TC-065", test_tc065_jwt_token_structure),
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
