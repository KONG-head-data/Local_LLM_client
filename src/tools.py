import json
from typing import Dict, Any, Callable, Union

# Dictionary to store executable functions
AVAILABLE_TOOLS: Dict[str, Callable] = {}

def tool(name: str):
    """Decorator to register a function as a tool."""
    def decorator(func: Callable):
        AVAILABLE_TOOLS[name] = func
        return func
    return decorator

# --- Define your tools here ---

@tool("get_weather")
def get_weather(location: str):
    """Mock tool to get weather."""
    return f"The weather in {location} is 25°C and sunny."

@tool("get_stock_price")
def get_stock_price(symbol: str):
    """Mock tool to get stock price."""
    prices = {"AAPL": 180.50, "NVDA": 135.20, "GOOGL": 142.10}
    price = prices.get(symbol.upper(), "Price not found")
    return f"The current price of {symbol} is ${price}."

@tool("calculate_sum")
def calculate_sum(a: int, b: int):
    """A simple tool to add two numbers."""
    return a + b

def execute_tool(tool_call: Union[str, Dict[str, Any]]) -> str:
    """
    Parses a tool call (string or dict) and executes it.
    If string, expected format: call:function_name{arg1:val1, arg2:val2}
    If dict, expected keys: 'name' and 'arguments'
    """
    try:
        if isinstance(tool_call, dict):
            name = tool_call.get("name")
            args = tool_call.get("arguments", {})
        else:
            call_part = tool_call.replace("call:", "").strip()
            
            name, args_str = call_part.split("{", 1)
            name = name.strip()
            
            args_str = args_str.rstrip("}").strip()
            
            args = {}
            if args_str:
                try:
                    # Attempt to convert to JSON-like string if not already
                    # Gemma template often outputs key:val instead of "key":"val"
                    cleaned_args = ""
                    for part in args_str.split(","):
                        if ":" in part:
                            k, v = part.split(":", 1)
                            # Handle potential quotes
                            k = k.strip().strip('"').strip("'")
                            v = v.strip().strip('"').strip("'")
                            cleaned_args += f'"{k}":"{v}",'
                    
                    if cleaned_args:
                        args = json.loads("{" + cleaned_args.rstrip(",") + "}")
                    else:
                        args = json.loads("{" + args_str + "}")
                except:
                    # Manual split as fallback
                    parts = args_str.split(",")
                    for p in parts:
                        if ":" in p:
                            k, v = p.split(":", 1)
                            args[k.strip().replace('"', "").replace("'", "")] = v.strip().replace('"', "").replace("'", "")

        if name in AVAILABLE_TOOLS:
            # Type conversion attempt for common types
            params = AVAILABLE_TOOLS[name].__annotations__
            typed_args = {}
            for k, v in args.items():
                if k in params:
                    try:
                        if params[k] == int: typed_args[k] = int(float(v))
                        elif params[k] == float: typed_args[k] = float(v)
                        elif params[k] == bool: typed_args[k] = str(v).lower() == "true"
                        else: typed_args[k] = v
                    except:
                        typed_args[k] = v
                else:
                    typed_args[k] = v
                    
            result = AVAILABLE_TOOLS[name](**typed_args)
            return str(result)
        else:
            return f"Error: Tool '{name}' not found."
            
    except Exception as e:
        return f"Error executing tool: {str(e)}"
