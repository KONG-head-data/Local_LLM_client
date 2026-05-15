import requests
import json

BASE_URL = "http://localhost:8888"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.json()}")

if __name__ == "__main__":
    # Note: The server must be running with MODEL_PATH set for these to work.
    print("Testing API (ensure server is running with MODEL_PATH)...")
    try:
        test_health()
    except Exception as e:
        print(f"Connection failed: {e}")
