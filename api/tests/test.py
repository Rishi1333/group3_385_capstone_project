import requests

url = "http://127.0.0.1:3000/predict/symptoms"

test_sentences = [
    "I have a high temperature, headache and my throat hurts",
    "Feeling weak and tired with some body pain",
    "No fever but I am coughing a lot",
    "My stomach hurts and I have nausea",
    "I am dizzy and sweating heavily",
    "Throwing up and can't eat anything",
    "I have chills and shivering since morning"
]

for text in test_sentences:
    data = {"text": text}
    response = requests.post(url, json=data)

    print(f"\nInput: {text}")
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
