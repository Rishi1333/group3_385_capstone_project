"""
Unit Tests for ClinicService Component

Test Cases: TC-012, TC-013, TC-014, TC-015, TC-016
Tests clinic recommendations and specialty mapping.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.clinic_service import ClinicService


def get_clinic_service():
    """Create a ClinicService instance for testing."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config",
        "clinics.json"
    )
    return ClinicService(clinics_config_path=config_path)


def test_tc012_get_specialties_headache():
    """TC-012: Get recommended specialties for headache"""
    print("\n=== TC-012: Get Specialties for Headache ===")
    
    clinic_service = get_clinic_service()
    specialties = clinic_service.get_recommended_specialties(["Headache"])
    
    # Assertions
    assert len(specialties) > 0, "Expected at least one specialty"
    assert "general" in specialties or "neurology" in specialties, f"Expected general or neurology, got {specialties}"
    
    print(f"Input: [\"Headache\"]")
    print(f"Result: {specialties}")
    print("[PASS] TC-012 passed")
    return True


def test_tc013_find_clinics_cardiology():
    """TC-013: Find clinics by cardiology specialty"""
    print("\n=== TC-013: Find Clinics by Cardiology ===")
    
    clinic_service = get_clinic_service()
    clinics = clinic_service.find_clinics_by_specialty(["cardiology"])
    
    # Assertions
    assert len(clinics) > 0, "Expected at least one clinic with cardiology"
    for clinic in clinics:
        assert "cardiology" in clinic.get("specialties", []), f"Clinic {clinic.get('id')} doesn't have cardiology"
    
    print(f"Input: [\"cardiology\"]")
    print(f"Result: Found {len(clinics)} clinics with cardiology")
    print("[PASS] TC-013 passed")
    return True


def test_tc014_empty_conditions():
    """TC-014: Handle empty conditions list"""
    print("\n=== TC-014: Empty Conditions Handling ===")
    
    clinic_service = get_clinic_service()
    specialties = clinic_service.get_recommended_specialties([])
    
    # Assertions
    assert "general" in specialties, f"Expected 'general' as fallback, got {specialties}"
    
    print(f"Input: []")
    print(f"Result: {specialties}")
    print("[PASS] TC-014 passed")
    return True


def test_tc015_invalid_clinic_id():
    """TC-015: Handle invalid clinic ID"""
    print("\n=== TC-015: Invalid Clinic ID Handling ===")
    
    clinic_service = get_clinic_service()
    result = clinic_service.get_clinic_by_id("nonexistent")
    
    # Assertions
    assert result is None, f"Expected None for invalid clinic ID, got {result}"
    
    print(f"Input: \"nonexistent\"")
    print(f"Result: {result}")
    print("[PASS] TC-015 passed")
    return True


def test_tc016_invalid_date():
    """TC-016: Handle invalid date format"""
    print("\n=== TC-016: Invalid Date Handling ===")
    
    clinic_service = get_clinic_service()
    slots = clinic_service.get_available_slots(
        "clinic-001",
        "doc-001",
        "invalid-date"
    )
    
    # Assertions
    assert slots == [], f"Expected empty list for invalid date, got {slots}"
    
    print(f"Input: \"invalid-date\"")
    print(f"Result: {slots}")
    print("[PASS] TC-016 passed")
    return True


def run_all_tests():
    """Run all ClinicService tests."""
    print("=" * 60)
    print("CLINIC SERVICE UNIT TESTS")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("TC-012", test_tc012_get_specialties_headache),
        ("TC-013", test_tc013_find_clinics_cardiology),
        ("TC-014", test_tc014_empty_conditions),
        ("TC-015", test_tc015_invalid_clinic_id),
        ("TC-016", test_tc016_invalid_date),
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
