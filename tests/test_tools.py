import requests
import json

BASE_URL = "http://localhost:8888"

def test_automatic_tool_execution():
    print("\n--- Testing Automatic Tool Execution (Agent Mode) ---")
    
    # Define the tools exactly as they are in src/tools.py
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"}
                    },
                    "required": ["symbol"]
                }
            }
        }
    ]
    
    # Complex query requiring two tools
    payload = {
        "messages": [
            {"role": "user", "content": "What is the weather on Mars and what is the current price of oil?"}
        ],
        "tools": tools,
        "thinking": True,
        "max_tokens": 2048
    }
    
    print("Sending agentic request (this may take a few seconds as it runs multiple loops)...")
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            message = data['message']
            history = data.get('history')
            
            print("\nFINAL ANSWER:")
            print("-" * 20)
            print(message['content'])
            print("-" * 20)
            
            # Verify if history contains tool results
            if history:
                tool_messages = [m for m in history if m.get('role') == 'tool']
                print(f"\n✅ Automatic Execution verified! {len(tool_messages)} tool calls were performed internally.")
                for tm in tool_messages:
                    print(f"  - Tool '{tm.get('name')}' returned: {tm.get('content')}")
            
            lower_content = message['content'].lower()
            if "tokyo" in lower_content and "nvda" in lower_content:
                print("\n✅ Success: The model successfully integrated both tool results into the final answer.")
            else:
                print("\nℹ️ Check: Ensure the tool names and cities match the output.")
                
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_automatic_tool_execution()
