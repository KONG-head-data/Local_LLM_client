import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.json()}")

def test_generate():
    payload = {
        "prompt": "Explain quantum physics in one sentence.",
        "max_tokens": 50,
        "temperature": 0.2
    }
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    if response.status_code == 200:
        print(f"Generate Response: {response.json()['text']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # Note: The server must be running with MODEL_PATH set for these to work.
    print("Testing API (ensure server is running with MODEL_PATH)...")
    try:
        test_health()
        # test_generate() # Uncomment when a model is loaded
    except Exception as e:
        print(f"Connection failed: {e}")
