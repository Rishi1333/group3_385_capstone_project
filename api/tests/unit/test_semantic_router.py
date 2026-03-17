"""
Unit Tests for SemanticRouter Component

Test Cases: TC-005, TC-006
Tests intent classification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.semantic_router import SemanticRouter, Intent


def test_tc005_classify_symptom_intent():
    """TC-005: Classify symptom triage intent"""
    print("\n=== TC-005: Classify Symptom Intent ===")
    
    router = SemanticRouter()
    result = router.classify_intent("I have a headache")
    
    # Assertions
    assert result["intent"] == Intent.SYMPTOM_TRIAGE, f"Expected SYMPTOM_TRIAGE, got {result['intent']}"
    
    print(f"Input: \"I have a headache\"")
    print(f"Result: intent={result['intent']}")
    print("[PASS] TC-005 passed")
    return True


def test_tc006_classify_image_intent():
    """TC-006: Classify image analysis intent with has_image=true"""
    print("\n=== TC-006: Classify Image Intent ===")
    
    router = SemanticRouter()
    result = router.classify_intent("Check this image", has_image=True)
    
    # Assertions
    assert result["intent"] == Intent.IMAGE_ANALYSIS, f"Expected IMAGE_ANALYSIS, got {result['intent']}"
    assert result["confidence"] == 1.0, f"Expected confidence 1.0, got {result['confidence']}"
    
    print(f"Input: \"Check this image\" with has_image=True")
    print(f"Result: intent={result['intent']}, confidence={result['confidence']}")
    print("[PASS] TC-006 passed")
    return True


def run_all_tests():
    """Run all SemanticRouter tests."""
    print("=" * 60)
    print("SEMANTIC ROUTER UNIT TESTS")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("TC-005", test_tc005_classify_symptom_intent),
        ("TC-006", test_tc006_classify_image_intent),
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
