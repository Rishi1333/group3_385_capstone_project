import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.feature_extractor import FeatureExtractor
from services.symptom_predictor import SymptomPredictor
from services.artifact_loader import ArtifactLoader

API_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = API_DIR.parent
ARTIFACT_PATH = str(PROJECT_ROOT / "artifacts")
KEYWORDS_FILE = str(API_DIR / "config" / "symptom_mapping.json")
model_loader = ArtifactLoader(ARTIFACT_PATH)
symptom_bundle = model_loader.load_symptom_bundle()
symptom_model = symptom_bundle["model"]
symptom_config = symptom_bundle["config"]
symptom_predictor = SymptomPredictor(symptom_model,symptom_config)
feature_extractor = FeatureExtractor(KEYWORDS_FILE,symptom_config)


# # Conversation session
session = {
    "collected_features": {f: 0 for f in symptom_config["features"]},
    "mode": "structured",  # or "fallback"
    "asked_features": []
}

def get_user_input():
    return input("You: ")

def ask_user(feature):
    answer = input(f"Chatbot: Do you have {feature.replace('_', ' ')}? (yes/no)\nYou: ")
    return 1 if answer.lower() in ["yes", "y"] else 0

def handle_input(user_text):
    global session

    if session["mode"] == "fallback":
        print("Chatbot: Tell me more about how you're feeling...")
        return

    # Step 1: Extract features from text
    detected = feature_extractor.extract(user_text)
    session["collected_features"].update(detected)

    # Step 2: Identify remaining features to ask
    remaining = [f for f, v in session["collected_features"].items()
                 if v == 0 and f not in session["asked_features"]]

    # Step 3: Ask about missing features
    for feature in remaining:
        val = ask_user(feature)
        session["collected_features"][feature] = val
        session["asked_features"].append(feature)

    # Step 4: Make prediction
    # Instead of converting to vector (list), send the dict
    pred = symptom_predictor.predict(session["collected_features"])
    print(f"Chatbot: Prediction: {pred['predictions']}")


    # Step 5: Switch to fallback for future inputs
    session["mode"] = "fallback"

# ---- Chat loop ----
print("Chatbot: Hi! How can I help you today?")
while True:
    user_input = get_user_input()
    if user_input.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye!")
        break
    handle_input(user_input)



# test_answers = {
#     "fever": 1, "cough": 1, "headache": 1, "nausea": 1,
#     "vomiting": 1, "fatigue": 0, "sore_throat": 0, "chills": 1,
#     "body_pain": 0, "loss_of_appetite": 0, "abdominal_pain": 1,
#     "diarrhea": 0, "sweating": 1, "rapid_breathing": 1, "dizziness": 0
# }

# for feature in symptom_config["features"]:
#     if session["collected_features"][feature] == 0:
#         session["collected_features"][feature] = test_answers[feature]

# pred = symptom_predictor.predict(session["collected_features"])
# print(f"Test Prediction: {pred['predictions']}")
