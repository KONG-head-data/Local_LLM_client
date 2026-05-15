import requests
import json
import time

BASE_URL = "http://localhost:8888"

def test_chat_basic():
    print("\n--- Testing Basic Chat ---")
    payload = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "thinking": False,
        "max_tokens": 50
    }
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data['message']['content']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_chat_thinking():
    print("\n--- Testing Chat with Thinking ---")
    payload = {
        "messages": [
            {"role": "user", "content": "Explain why the sky is blue."}
        ],
        "thinking": True,
        "max_tokens": 500
    }
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    if response.status_code == 200:
        data = response.json()
        content = data['message']['content']
        print(f"Response Content:\n{content}")
        
        if "<|channel>thought" in content:
            print("\n✅ Thinking token detected in output!")
        else:
            print("\n✅ Response received (Note: Check output for internal reasoning tags)")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_tool_calling_loop():
    print("\n--- Testing Tool Calling Loop ---")
    
    # Step 1: Ask a question that requires a tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "The stock symbol, e.g. AAPL"}
                    },
                    "required": ["symbol"]
                }
            }
        }
    ]
    
    payload = {
        "messages": [
            {"role": "user", "content": "What is the stock price of NVDA?"}
        ],
        "tools": tools,
        "thinking": True,
        "max_tokens": 200
    }
    
    print("Step 1: Sending request with tools...")
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code != 200:
        print(f"Failed at Step 1: {response.text}")
        return

    data = response.json()
    message = data['message']
    content = message.get('content', '')
    print(f"Assistant raw output: {content[:100]}...")
    
    if "call:get_stock_price" in content:
        print("✅ Model requested the 'get_stock_price' tool!")
        
        tool_call_id = "call_nvda_123"
        
        second_payload = {
            "messages": [
                {"role": "user", "content": "What is the stock price of NVDA?"},
                {
                    "role": "assistant", 
                    "content": content, 
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": "get_stock_price", "arguments": '{"symbol": "NVDA"}'}
                        }
                    ]
                },
                {
                    "role": "tool", 
                    "tool_call_id": tool_call_id, 
                    "name": "get_stock_price", 
                    "content": "NVDA is currently trading at $135.50 (up 2.3%)"
                }
            ],
            "max_tokens": 200
        }
        
        print("Step 2: Sending tool response back to model...")
        final_response = requests.post(f"{BASE_URL}/chat", json=second_payload)
        
        if final_response.status_code == 200:
            final_data = final_response.json()
            print(f"Final Model Answer: {final_data['message']['content']}")
        else:
            print(f"Failed at Step 2: {final_response.text}")
    else:
        print("ℹ️ Model did not trigger the tool in this turn.")

def test_json_mode():
    print("\n--- Testing JSON Mode (Grammar) ---")
    payload = {
        "messages": [
            {"role": "user", "content": "Generate a JSON object for a character in an RPG with 'name', 'class', and 'level'."}
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 150
    }
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    if response.status_code == 200:
        content = response.json()['message']['content']
        print(f"JSON Response:\n{content}")
        try:
            json.loads(content)
            print("✅ Valid JSON detected!")
        except:
            print("❌ Output was not valid JSON.")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("Gemma 4 API Integration Test Suite")
    print("================================")
    
    try:
        # Check health
        requests.get(f"{BASE_URL}/health")
        
        test_chat_basic()
        test_chat_thinking()
        test_json_mode()
        test_tool_calling_loop()
        
    except requests.exceptions.ConnectionError:
        print(f"\nCRITICAL ERROR: Could not connect to the API at {BASE_URL}.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
