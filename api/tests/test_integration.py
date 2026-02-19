"""
Integration test script for AI Virtual Clinic API
Tests all model endpoints and routing functionality.
"""
import requests
import json

BASE_URL = "http://localhost:3000"

def test_health():
    """Test health endpoint."""
    print("\n=== Testing /health ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_models():
    """Test models endpoint."""
    print("\n=== Testing /models ===")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Available models: {len(data['models'])}")
    for model in data['models']:
        print(f"  - {model['type']}: {model['name']}")
    return response.status_code == 200

def test_routing():
    """Test routing endpoint with different inputs."""
    print("\n=== Testing /predict/routing ===")
    
    test_cases = [
        ("I have chest pain and shortness of breath", "heart"),
        ("I have high blood sugar and frequent urination", "diabetes"),
        ("I feel depressed and anxious all the time", "mental_health"),
        ("I have fever, cough, and headache", "symptom_disease"),
    ]
    
    results = []
    for text, expected in test_cases:
        response = requests.post(
            f"{BASE_URL}/predict/routing",
            json={"text": text}
        )
        if response.status_code == 200:
            data = response.json()
            model_type = data.get("model_type", "unknown")
            confidence = data.get("confidence", 0)
            matched = data.get("matched_keywords", {})
            
            status = "[PASS]" if model_type == expected else "[FAIL]"
            print(f"{status} '{text[:40]}...' -> {model_type} (expected: {expected}, confidence: {confidence:.2%})")
            if matched:
                for model, keywords in matched.items():
                    if keywords:
                        print(f"   Matched {model}: {keywords[:3]}")
            results.append(model_type == expected)
        else:
            print(f"[FAIL] Error: {response.status_code}")
            results.append(False)
    
    return all(results)

def test_direct_prediction():
    """Test direct prediction endpoint."""
    print("\n=== Testing /predict/direct ===")
    
    # Test diabetes prediction
    print("\nDiabetes Prediction:")
    response = requests.post(
        f"{BASE_URL}/predict/direct",
        json={
            "model_type": "diabetes",
            "features": {
                "Pregnancies": 2,
                "Glucose": 150,
                "BloodPressure": 80,
                "SkinThickness": 25,
                "Insulin": 100,
                "BMI": 32.5,
                "DiabetesPedigreeFunction": 0.5,
                "Age": 45
            }
        }
    )
    if response.status_code == 200:
        data = response.json()
        print(f"  Prediction: {data['prediction']['prediction']}")
        print(f"  Probability: {data['prediction']['probability']:.2%}")
    else:
        print(f"  Error: {response.status_code}")
    
    # Test symptom-disease extended prediction
    print("\nSymptom-Disease Extended Prediction:")
    response = requests.post(
        f"{BASE_URL}/predict/direct",
        json={
            "model_type": "symptom_disease_extended",
            "text": "I have severe headache, fever, and sensitivity to light"
        }
    )
    if response.status_code == 200:
        data = response.json()
        print(f"  Prediction: {data['prediction']['prediction']}")
        print(f"  Confidence: {data['prediction']['confidence']:.2%}")
        print(f"  Top 3 probabilities: {data['prediction']['probabilities'][:3]}")
    else:
        print(f"  Error: {response.status_code}")
    
    return True

def test_symptom_prediction():
    """Test symptom prediction endpoint."""
    print("\n=== Testing /predict/symptoms/start ===")
    
    response = requests.post(
        f"{BASE_URL}/predict/symptoms/start",
        json={"text": "I have fever and headache"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"  Status: {data['status']}")
        print(f"  Model Type: {data.get('model_type', 'N/A')}")
        if 'prediction' in data:
            print(f"  Prediction: {data['prediction']}")
        if 'questions' in data:
            print(f"  Questions: {len(data['questions'])} questions generated")
        return True
    else:
        print(f"  Error: {response.status_code}")
        print(f"  Response: {response.text}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("AI Virtual Clinic API - Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health),
        ("Models List", test_models),
        ("Routing", test_routing),
        ("Direct Prediction", test_direct_prediction),
        ("Symptom Prediction", test_symptom_prediction),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n[FAIL] {name} failed with error: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        print(f"  {name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all(results.values())

if __name__ == "__main__":
    main()
