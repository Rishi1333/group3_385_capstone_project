"""
Unit Tests for SafetyGuardrails Component

Test Cases: TC-001, TC-002, TC-003, TC-004
Tests emergency detection and severity classification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.safety_guardrails import SafetyGuardrails


def test_tc001_detect_critical_emergency():
    """TC-001: Detect critical emergency - chest pain and breathing issues"""
    print("\n=== TC-001: Detect Critical Emergency ===")
    
    guardrails = SafetyGuardrails()
    result = guardrails._fallback_detection("I'm having chest pain and can't breathe")
    
    # Assertions
    assert result["is_emergency"] is True, "Expected is_emergency to be True"
    assert result["severity"] == SafetyGuardrails.SEVERITY_CRITICAL, f"Expected CRITICAL severity, got {result['severity']}"
    
    print(f"Input: \"I'm having chest pain and can't breathe\"")
    print(f"Result: is_emergency={result['is_emergency']}, severity={result['severity']}")
    print("[PASS] TC-001 passed")
    return True


def test_tc002_detect_high_urgency():
    """TC-002: Detect high urgency - severe stomach pain"""
    print("\n=== TC-002: Detect High Urgency ===")
    
    guardrails = SafetyGuardrails()
    result = guardrails._fallback_detection("I have severe stomach pain")
    
    # Assertions
    assert result["is_emergency"] is True, "Expected is_emergency to be True"
    assert result["severity"] == SafetyGuardrails.SEVERITY_HIGH, f"Expected HIGH severity, got {result['severity']}"
    
    print(f"Input: \"I have severe stomach pain\"")
    print(f"Result: is_emergency={result['is_emergency']}, severity={result['severity']}")
    print("[PASS] TC-002 passed")
    return True


def test_tc003_detect_low_severity():
    """TC-003: Detect low severity - mild headache"""
    print("\n=== TC-003: Detect Low Severity ===")
    
    guardrails = SafetyGuardrails()
    result = guardrails._fallback_detection("I have a mild headache")
    
    # Assertions
    assert result["is_emergency"] is False, "Expected is_emergency to be False"
    assert result["severity"] == SafetyGuardrails.SEVERITY_LOW, f"Expected LOW severity, got {result['severity']}"
    
    print(f"Input: \"I have a mild headache\"")
    print(f"Result: is_emergency={result['is_emergency']}, severity={result['severity']}")
    print("[PASS] TC-003 passed")
    return True


def test_tc004_empty_input():
    """TC-004: Handle empty input gracefully"""
    print("\n=== TC-004: Empty Input Handling ===")
    
    guardrails = SafetyGuardrails()
    result = guardrails._fallback_detection("")
    
    # Assertions
    assert result["is_emergency"] is False, "Expected is_emergency to be False for empty input"
    assert result["severity"] == SafetyGuardrails.SEVERITY_LOW, f"Expected LOW severity for empty input, got {result['severity']}"
    
    print(f"Input: \"\"")
    print(f"Result: is_emergency={result['is_emergency']}, severity={result['severity']}")
    print("[PASS] TC-004 passed")
    return True


def run_all_tests():
    """Run all SafetyGuardrails tests."""
    print("=" * 60)
    print("SAFETY GUARDRAILS UNIT TESTS")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("TC-001", test_tc001_detect_critical_emergency()))
    except AssertionError as e:
        print(f"[FAIL] TC-001: {e}")
        results.append(("TC-001", False))
    
    try:
        results.append(("TC-002", test_tc002_detect_high_urgency()))
    except AssertionError as e:
        print(f"[FAIL] TC-002: {e}")
        results.append(("TC-002", False))
    
    try:
        results.append(("TC-003", test_tc003_detect_low_severity()))
    except AssertionError as e:
        print(f"[FAIL] TC-003: {e}")
        results.append(("TC-003", False))
    
    try:
        results.append(("TC-004", test_tc004_empty_input()))
    except AssertionError as e:
        print(f"[FAIL] TC-004: {e}")
        results.append(("TC-004", False))
    
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
