"""
Unit Tests for ClinicalReportGenerator Component

Test Cases: TC-048, TC-049, TC-050, TC-051
Tests clinical report generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.clinical_report import ClinicalReportGenerator


def test_tc048_get_disclaimer():
    """TC-048: Get disclaimer mentions not a medical diagnosis"""
    print("\n=== TC-048: Get Disclaimer ===")
    
    generator = ClinicalReportGenerator()
    disclaimer = generator._get_disclaimer()
    
    # Assertions
    assert "NOT a medical diagnosis" in disclaimer, "Disclaimer should mention 'NOT a medical diagnosis'"
    assert "AI" in disclaimer, "Disclaimer should mention 'AI'"
    
    print(f"Result: Disclaimer contains required text")
    print("[PASS] TC-048 passed")
    return True


def test_tc049_generate_report():
    """TC-049: Generate report with all sections"""
    print("\n=== TC-049: Generate Report ===")
    
    generator = ClinicalReportGenerator()
    
    clinical_data = {
        "patient_profile": {"age": 45, "sex": "male"},
        "chief_complaint": "Headache",
        "symptoms": [{"name": "headache"}],
        "risk_factors": ["stress"]
    }
    
    report = generator._generate_fallback_report(clinical_data, [])
    
    # Assertions
    assert report is not None, "Report should be generated"
    assert report["chief_complaint"] == "Headache", "Chief complaint should match"
    assert "disclaimer" in report, "Report should have disclaimer"
    
    print(f"Input: Clinical data with headache")
    print(f"Result: Report generated with chief_complaint={report['chief_complaint']}")
    print("[PASS] TC-049 passed")
    return True


def test_tc050_format_for_display():
    """TC-050: Format report for display"""
    print("\n=== TC-050: Format For Display ===")
    
    generator = ClinicalReportGenerator()
    
    report = {
        "report_id": "RPT-001",
        "generated_at": "2024-01-01T00:00:00",
        "patient_profile": {"age": 45, "sex": "male"},
        "chief_complaint": "Headache",
        "history_of_present_illness": "Patient reports headache",
        "risk_factors": ["stress"],
        "suggested_differential": [{"condition": "Tension headache", "likelihood": "High"}],
        "recommended_next_steps": ["Rest", "Hydration"],
        "disclaimer": "Test disclaimer"
    }
    
    formatted = generator.format_for_display(report)
    
    # Assertions
    assert "CLINICAL SUMMARY REPORT" in formatted, "Should contain header"
    assert "Headache" in formatted, "Should contain chief complaint"
    
    print(f"Result: Formatted string contains 'CLINICAL SUMMARY REPORT' and 'Headache'")
    print("[PASS] TC-050 passed")
    return True


def test_tc051_empty_symptoms():
    """TC-051: Generate report with empty symptoms"""
    print("\n=== TC-051: Empty Symptoms Handling ===")
    
    generator = ClinicalReportGenerator()
    
    clinical_data = {
        "patient_profile": {"age": 30, "sex": "female"},
        "chief_complaint": "Fatigue",
        "symptoms": [],
        "risk_factors": []
    }
    
    report = generator._generate_fallback_report(clinical_data, [])
    
    # Assertions
    assert report is not None, "Report should still be generated"
    assert report["chief_complaint"] == "Fatigue", "Chief complaint should match"
    
    print(f"Input: Empty symptoms list")
    print(f"Result: Report generated successfully")
    print("[PASS] TC-051 passed")
    return True


def run_all_tests():
    """Run all ClinicalReportGenerator tests."""
    print("=" * 60)
    print("CLINICAL REPORT GENERATOR UNIT TESTS")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("TC-048", test_tc048_get_disclaimer),
        ("TC-049", test_tc049_generate_report),
        ("TC-050", test_tc050_format_for_display),
        ("TC-051", test_tc051_empty_symptoms),
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
