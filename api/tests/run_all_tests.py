"""
Test Runner - Runs all test scripts and updates test_cases.csv

This script:
1. Runs all unit tests
2. Runs all integration tests
3. Updates the test_cases.csv with results
"""

import sys
import os
import csv
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test results storage
test_results = {}


def run_unit_tests():
    """Run all unit tests."""
    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    # Import and run SafetyGuardrails tests
    try:
        from tests.unit.test_safety_guardrails import (
            test_tc001_detect_critical_emergency,
            test_tc002_detect_high_urgency,
            test_tc003_detect_low_severity,
            test_tc004_empty_input
        )
        
        tests = [
            ("TC-001", test_tc001_detect_critical_emergency),
            ("TC-002", test_tc002_detect_high_urgency),
            ("TC-003", test_tc003_detect_low_severity),
            ("TC-004", test_tc004_empty_input),
        ]
        
        for test_id, test_func in tests:
            try:
                test_func()
                test_results[test_id] = "Pass"
            except AssertionError as e:
                print(f"[FAIL] {test_id}: {e}")
                test_results[test_id] = "Fail"
            except Exception as e:
                print(f"[ERROR] {test_id}: {e}")
                test_results[test_id] = "Error"
    except ImportError as e:
        print(f"Could not import safety guardrails tests: {e}")
    
    # Import and run ClinicService tests
    try:
        from tests.unit.test_clinic_service_unit import (
            test_tc012_get_specialties_headache,
            test_tc013_find_clinics_cardiology,
            test_tc014_empty_conditions,
            test_tc015_invalid_clinic_id,
            test_tc016_invalid_date
        )
        
        tests = [
            ("TC-012", test_tc012_get_specialties_headache),
            ("TC-013", test_tc013_find_clinics_cardiology),
            ("TC-014", test_tc014_empty_conditions),
            ("TC-015", test_tc015_invalid_clinic_id),
            ("TC-016", test_tc016_invalid_date),
        ]
        
        for test_id, test_func in tests:
            try:
                test_func()
                test_results[test_id] = "Pass"
            except AssertionError as e:
                print(f"[FAIL] {test_id}: {e}")
                test_results[test_id] = "Fail"
            except Exception as e:
                print(f"[ERROR] {test_id}: {e}")
                test_results[test_id] = "Error"
    except ImportError as e:
        print(f"Could not import clinic service tests: {e}")
    
    # Import and run SessionStore tests
    try:
        from tests.unit.test_session_store_unit import (
            test_tc010_create_session,
            test_tc011_get_nonexistent_session
        )
        
        tests = [
            ("TC-010", test_tc010_create_session),
            ("TC-011", test_tc011_get_nonexistent_session),
        ]
        
        for test_id, test_func in tests:
            try:
                test_func()
                test_results[test_id] = "Pass"
            except AssertionError as e:
                print(f"[FAIL] {test_id}: {e}")
                test_results[test_id] = "Fail"
            except Exception as e:
                print(f"[ERROR] {test_id}: {e}")
                test_results[test_id] = "Error"
    except ImportError as e:
        print(f"Could not import session store tests: {e}")


def run_integration_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60)
    
    # Import and run Booking API tests
    try:
        from tests.integration.test_booking_api import (
            test_tc017_recommend_endpoint,
            test_tc018_recommend_missing_conditions,
            test_tc019_clinics_endpoint,
            test_tc020_clinic_not_found
        )
        
        tests = [
            ("TC-017", test_tc017_recommend_endpoint),
            ("TC-018", test_tc018_recommend_missing_conditions),
            ("TC-019", test_tc019_clinics_endpoint),
            ("TC-020", test_tc020_clinic_not_found),
        ]
        
        for test_id, test_func in tests:
            try:
                test_func()
                test_results[test_id] = "Pass"
            except AssertionError as e:
                print(f"[FAIL] {test_id}: {e}")
                test_results[test_id] = "Fail"
            except Exception as e:
                print(f"[ERROR] {test_id}: {e}")
                test_results[test_id] = "Error"
    except ImportError as e:
        print(f"Could not import booking API tests: {e}")


def update_csv():
    """Update test_cases.csv with results."""
    print("\n" + "=" * 60)
    print("UPDATING TEST_CASES.CSV")
    print("=" * 60)
    
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "docs",
        "test_cases.csv"
    )
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return
    
    # Read existing CSV
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            test_id = row.get('Test case ID', '')
            if test_id in test_results:
                row['Execution status'] = test_results[test_id]
                row['Actual results'] = f"Test {test_results[test_id].lower()}ed"
                row['Test case executer'] = "Automated Test Runner"
            rows.append(row)
    
    # Write updated CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Updated {len(test_results)} test results in {csv_path}")


def print_summary():
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in test_results.values() if r == "Pass")
    failed = sum(1 for r in test_results.values() if r == "Fail")
    error = sum(1 for r in test_results.values() if r == "Error")
    total = len(test_results)
    
    print(f"\nTotal tests run: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Errors: {error}")
    
    if total > 0:
        print(f"\nPass rate: {passed/total*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_id, result in sorted(test_results.items()):
        status = "[PASS]" if result == "Pass" else "[FAIL]"
        print(f"  {status} {test_id}: {result}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("AI VIRTUAL CLINIC - AUTOMATED TEST RUNNER")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Run tests
    run_unit_tests()
    run_integration_tests()
    
    # Update CSV
    update_csv()
    
    # Print summary
    print_summary()
    
    # Return exit code
    failed = sum(1 for r in test_results.values() if r in ["Fail", "Error"])
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
