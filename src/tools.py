import json
from typing import Dict, Any, Callable

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

def execute_tool(tool_call_str: str) -> str:
    """
    Parses a tool call string and executes it.
    Expected format: call:function_name{arg1:val1, arg2:val2}
    """
    try:
        call_part = tool_call_str.replace("call:", "").strip()
        
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
                        cleaned_args += f'"{k.strip()}":"{v.strip()}",'
                
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
