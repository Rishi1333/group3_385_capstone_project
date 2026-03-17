"""
Unit Tests for SessionStore Component

Test Cases: TC-010, TC-011
Tests session creation and retrieval.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.session_store import SessionStore


def test_tc010_create_session():
    """TC-010: Create session and verify UUID format"""
    print("\n=== TC-010: Create Session ===")
    
    store = SessionStore()
    session_id = store.create({"test": "data"})
    
    # Assertions
    assert session_id is not None, "Expected session_id to be created"
    assert len(session_id) == 36, f"Expected UUID format (36 chars), got {len(session_id)} chars"
    assert session_id.count('-') == 4, f"Expected UUID format with 4 hyphens, got {session_id.count('-')}"
    
    print(f"Result: Session created with ID: {session_id}")
    print("[PASS] TC-010 passed")
    return True


def test_tc011_get_nonexistent_session():
    """TC-011: Get nonexistent session returns None"""
    print("\n=== TC-011: Get Nonexistent Session ===")
    
    store = SessionStore()
    data = store.get("nonexistent-id-12345")
    
    # Assertions
    assert data is None, f"Expected None for nonexistent session, got {data}"
    
    print(f"Input: \"nonexistent-id-12345\"")
    print(f"Result: {data}")
    print("[PASS] TC-011 passed")
    return True


def run_all_tests():
    """Run all SessionStore tests."""
    print("=" * 60)
    print("SESSION STORE UNIT TESTS")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("TC-010", test_tc010_create_session),
        ("TC-011", test_tc011_get_nonexistent_session),
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
