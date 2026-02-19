"""
Test patient inputs for verifying model routing.
Run this script to test that the system correctly routes different patient inputs.
"""

import requests
import json

API_BASE = "http://127.0.0.1:3000"

# Test patient inputs for each model
TEST_INPUTS = [
    # Heart Model Tests
    {
        "input": "I have chest pain and shortness of breath",
        "expected_model": "heart",
        "description": "Classic heart symptoms"
    },
    {
        "input": "I'm worried about my heart health, I have high blood pressure",
        "expected_model": "heart",
        "description": "Heart health concern with hypertension"
    },
    {
        "input": "I feel tightness in my chest when I exercise",
        "expected_model": "heart",
        "description": "Exercise-induced chest tightness"
    },
    
    # Diabetes Model Tests
    {
        "input": "I have high blood sugar and frequent urination",
        "expected_model": "diabetes",
        "description": "Classic diabetes symptoms"
    },
    {
        "input": "I'm always thirsty and have blurry vision, could it be diabetes?",
        "expected_model": "diabetes",
        "description": "Diabetes concern with thirst and vision"
    },
    {
        "input": "My glucose levels have been high lately",
        "expected_model": "diabetes",
        "description": "High glucose mention"
    },
    
    # Mental Health Model Tests
    {
        "input": "I feel depressed and anxious all the time",
        "expected_model": "mental_health",
        "description": "Depression and anxiety"
    },
    {
        "input": "I've been feeling very stressed and having mood swings",
        "expected_model": "mental_health",
        "description": "Stress and mood swings"
    },
    {
        "input": "I'm having trouble sleeping and feel overwhelmed",
        "expected_model": "mental_health",
        "description": "Sleep issues and overwhelm"
    },
    
    # Symptom Disease Extended (NLP) Tests
    {
        "input": "I have fever, cough, and headache",
        "expected_model": "symptom_disease",
        "description": "General symptoms - NLP model"
    },
    {
        "input": "I have a runny nose and sore throat",
        "expected_model": "symptom_disease",
        "description": "Cold symptoms"
    },
    {
        "input": "I have nausea and stomach pain",
        "expected_model": "symptom_disease",
        "description": "Digestive symptoms"
    },
    
    # General Symptom Model Tests
    {
        "input": "I have itching and skin rash",
        "expected_model": "symptom",
        "description": "Skin symptoms"
    },
    {
        "input": "I have joint pain and fatigue",
        "expected_model": "symptom",
        "description": "General symptoms"
    },
]

def test_routing():
    """Test model routing with various patient inputs."""
    print("=" * 70)
    print("PATIENT INPUT ROUTING TESTS")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(TEST_INPUTS, 1):
        print(f"\n--- Test {i}: {test['description']} ---")
        print(f"Input: \"{test['input']}\"")
        print(f"Expected Model: {test['expected_model']}")
        
        try:
            response = requests.post(
                f"{API_BASE}/predict/routing",
                json={"text": test["input"]},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                detected_model = data.get("model_type", "unknown")
                confidence = data.get("confidence", 0)
                matched_keywords = data.get("matched_keywords", {})
                
                # Check if expected model matches detected
                is_correct = detected_model == test["expected_model"]
                
                status = "PASS" if is_correct else "FAIL"
                if is_correct:
                    passed += 1
                else:
                    failed += 1
                
                print(f"Detected Model: {detected_model} (confidence: {confidence:.1%})")
                print(f"Status: [{status}]")
                
                # Show matched keywords
                for model, keywords in matched_keywords.items():
                    if keywords:
                        print(f"  Matched {model}: {keywords}")
            else:
                print(f"ERROR: Status {response.status_code}")
                failed += 1
                
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{len(TEST_INPUTS)} tests passed")
    print("=" * 70)
    
    return passed, failed


def test_direct_predictions():
    """Test direct predictions for each model."""
    print("\n" + "=" * 70)
    print("DIRECT PREDICTION TESTS")
    print("=" * 70)
    
    # Test diabetes direct prediction
    print("\n--- Diabetes Prediction ---")
    diabetes_features = {
        "Pregnancies": 2,
        "Glucose": 150,
        "BloodPressure": 80,
        "SkinThickness": 25,
        "Insulin": 100,
        "BMI": 32.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/predict/direct",
            json={"model_type": "diabetes", "features": diabetes_features},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Prediction: {data.get('prediction')}")
            print(f"Probability: {data.get('probability', 0):.1%}")
        else:
            print(f"ERROR: {response.status_code}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test heart direct prediction
    print("\n--- Heart Prediction ---")
    heart_features = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 140,
        "chol": 240,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 1,
        "oldpeak": 2.5,
        "slope": 2,
        "ca": 1,
        "thal": 2
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/predict/direct",
            json={"model_type": "heart", "features": heart_features},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Prediction: {data.get('prediction')}")
            print(f"Probability: {data.get('probability', 0):.1%}")
        else:
            print(f"ERROR: {response.status_code}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test symptom-disease extended (NLP) prediction
    print("\n--- Symptom-Disease Extended (NLP) Prediction ---")
    try:
        response = requests.post(
            f"{API_BASE}/predict/direct",
            json={"model_type": "symptom_disease_extended", "text": "I have fever, cough, and body aches"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Prediction: {data.get('prediction')}")
            print(f"Confidence: {data.get('confidence', 0):.1%}")
            if data.get('probabilities'):
                print("Top 3 predictions:")
                for p in data['probabilities'][:3]:
                    print(f"  - {p[0]}: {p[1]:.1%}")
        else:
            print(f"ERROR: {response.status_code}")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    test_routing()
    test_direct_predictions()