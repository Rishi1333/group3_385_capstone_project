import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.feature_extractor import FeatureExtractor
from services.symptom_predictor import SymptomPredictor
from services.artifact_loader import ArtifactLoader
from services.llm_conversation import LLMConversation


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



session = {
    "collected_features": {f: 0 for f in symptom_config["features"]},
    "mode": "structured", 
    "asked_features": []
}

def get_user_input():
    return input("You: ")

def ask_user(feature):
    answer = input(f"Chatbot: Do you have {feature.replace('_', ' ')}? (yes/no)\nYou: ")

    return 1 if answer.lower() in ["yes", "y"] else 0

llm = LLMConversation()

def handle_input(user_text):
    global session

    if session["mode"] == "fallback":
        print("Chatbot: Tell me more about how you're feeling...")
        return

    detected = feature_extractor.extract(user_text)
    session["collected_features"].update(detected)

    remaining = [f for f, v in session["collected_features"].items()
                 if v == 0 and f not in session["asked_features"]]

    for feature in remaining:
        question = llm.next_question(feature, session["collected_features"])
        answer = input(f"Chatbot : {question}\nYou: ")

        val = 1 if answer.lower() in ["yes","y"] else 0
        session["collected_features"][feature] = val
        session["asked_features"].append(feature)

    pred = symptom_predictor.predict(session["collected_features"])
    print(f"Prediction: {pred['predictions']}")

    explanation = llm.explain_prediction(pred["predictions"], session["collected_features"])
    print(explanation)
    
    session["mode"] = "fallback"

print("Chatbot: Hi! How can I help you today?")
while True:
    user_input = get_user_input()
    if user_input.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye!")
        break
    handle_input(user_input)


