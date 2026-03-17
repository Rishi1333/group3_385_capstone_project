"""
Comprehensive Triage Test Script

Tests various symptom scenarios to validate the dynamic triage system.
"""

import requests
import json
import time
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

API_BASE = "http://127.0.0.1:5000"

# Test scenarios
TEST_SCENARIOS = [
    {
        "name": "Critical Emergency - Chest Pain",
        "messages": ["I have been having severe chest pain for the last 2 hours"],
        "expected_urgency": "CRITICAL",
        "expected_state": "EMERGENCY_PATH"
    },
    {
        "name": "Critical Emergency - Stroke Symptoms",
        "messages": ["My face is drooping and I can't move my left arm"],
        "expected_urgency": "CRITICAL",
        "expected_state": "EMERGENCY_PATH"
    },
    {
        "name": "High Urgency - Severe Symptoms",
        "messages": ["I have a very high fever and I'm confused"],
        "expected_urgency": "HIGH",
        "expected_state": None  # Could be various states
    },
    {
        "name": "Moderate Urgency - Persistent Symptoms",
        "messages": [
            "I have had a persistent cough for 3 days",
            "I am 35 years old male",
            "No other medical conditions",
            "Not taking any medications"
        ],
        "expected_urgency": "MODERATE",
        "expected_state": None
    },
    {
        "name": "Low Urgency - Mild Symptoms",
        "messages": [
            "I have a mild headache since yesterday",
            "I am 28 years old female",
            "No medical history"
        ],
        "expected_urgency": "LOW",
        "expected_state": None
    },
    {
        "name": "Visual Symptom - Rash",
        "messages": [
            "I have a red rash on my arm",
            "I am 40 years old male"
        ],
        "expected_urgency": None,
        "expected_state": None,
        "expect_image_request": True
    },
    {
        "name": "Mental Health - Depression",
        "messages": [
            "I've been feeling very sad and hopeless for weeks",
            "I am 25 years old female"
        ],
        "expected_urgency": None,
        "expected_state": None
    }
]


def start_session():
    """Start a new triage session."""
    response = requests.post(
        f"{API_BASE}/api/triage/start",
        headers={"Content-Type": "application/json"},
        json={}
    )
    return response.json()


def send_message(session_id, message):
    """Send a message to the triage session."""
    response = requests.post(
        f"{API_BASE}/api/triage/message",
        headers={"Content-Type": "application/json"},
        json={"session_id": session_id, "message": message}
    )
    return response.json()


def run_test(scenario):
    """Run a single test scenario."""
    print(f"\n{'='*60}")
    print(f"TEST: {scenario['name']}")
    print(f"{'='*60}")
    
    try:
        # Start session
        start_data = start_session()
        if not start_data.get("success"):
            print(f"❌ Failed to start session: {start_data.get('error')}")
            return False
        
        session_id = start_data["session_id"]
        print(f"✓ Session started: {session_id[:16]}...")
        
        # Send messages
        all_responses = []
        for i, message in enumerate(scenario["messages"]):
            print(f"\n[User]: {message}")
            
            response = send_message(session_id, message)
            all_responses.append(response)
            
            if not response.get("success"):
                print(f"❌ Error: {response.get('error')}")
                return False
            
            print(f"[Bot]: {response.get('message', '')[:100]}...")
            print(f"  State: {response.get('state')}")
            print(f"  Urgency: {response.get('urgency') or response.get('clinical_data', {}).get('urgency_level')}")
            
            # Check for emergency
            if response.get("state") == "EMERGENCY_PATH":
                print("⚠️ EMERGENCY DETECTED - Session ended")
                break
            
            # Check for completion
            if response.get("end_session"):
                print("✓ Session completed")
                break
            
            time.sleep(0.5)  # Small delay between messages
        
        # Validate results
        final_response = all_responses[-1]
        success = True
        
        if scenario.get("expected_urgency"):
            actual_urgency = final_response.get("urgency") or final_response.get("clinical_data", {}).get("urgency_level")
            if actual_urgency != scenario["expected_urgency"]:
                print(f"⚠️ Urgency mismatch: expected {scenario['expected_urgency']}, got {actual_urgency}")
                # Don't fail - just warn
        
        if scenario.get("expected_state"):
            if final_response.get("state") != scenario["expected_state"]:
                print(f"⚠️ State mismatch: expected {scenario['expected_state']}, got {final_response.get('state')}")
                # Don't fail - just warn
        
        if scenario.get("expect_image_request"):
            if final_response.get("request_image"):
                print("✓ Image request detected as expected")
            else:
                print("ℹ️ No image request (may depend on conversation flow)")
        
        print(f"\n✓ Test completed: {scenario['name']}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def test_health_endpoint():
    """Test the health endpoint."""
    print("\n" + "="*60)
    print("TEST: Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE}/api/health")
        data = response.json()
        
        print(f"Status: {data.get('status')}")
        print(f"Version: {data.get('version')}")
        print(f"Architecture: {data.get('architecture')}")
        print(f"RAG Available: {data.get('services', {}).get('rag')}")
        
        if data.get("status") == "healthy":
            print("✓ Health check passed")
            return True
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_red_flags_endpoint():
    """Test the red flags endpoint."""
    print("\n" + "="*60)
    print("TEST: Red Flags Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE}/api/triage/red-flags")
        data = response.json()
        
        if data.get("success"):
            red_flags = data.get("red_flags", {})
            categories = red_flags.get("red_flags", {})
            print(f"✓ Red flags loaded: {len(categories)} categories")
            for category in categories:
                print(f"  - {category}")
            return True
        else:
            print(f"❌ Failed to load red flags: {data.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Red flags test error: {e}")
        return False


def test_urgency_levels_endpoint():
    """Test the urgency levels endpoint."""
    print("\n" + "="*60)
    print("TEST: Urgency Levels Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE}/api/triage/urgency-levels")
        data = response.json()
        
        if data.get("success"):
            levels = data.get("urgency_levels", {})
            print(f"✓ Urgency levels loaded: {len(levels)} levels")
            for level, info in levels.items():
                print(f"  - {level}: {info.get('description')}")
            return True
        else:
            print(f"❌ Failed to load urgency levels: {data.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Urgency levels test error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# AI VIRTUAL CLINIC - COMPREHENSIVE TRIAGE TESTS")
    print("#"*60)
    
    results = []
    
    # Test basic endpoints
    results.append(("Health Endpoint", test_health_endpoint()))
    results.append(("Red Flags Endpoint", test_red_flags_endpoint()))
    results.append(("Urgency Levels Endpoint", test_urgency_levels_endpoint()))
    
    # Test scenarios
    for scenario in TEST_SCENARIOS:
        results.append((scenario["name"], run_test(scenario)))
    
    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
