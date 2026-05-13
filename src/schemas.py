import json
import re
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    stream: bool = False
    thinking: bool = False
    response_format: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]] # Support for multimodal content parts
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None # For tool response role


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None # OpenAI-style tool definitions
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    stream: bool = False
    thinking: bool = False
    response_format: Optional[Dict[str, Any]] = None

    def format_for_log(self) -> str:
        formatted = ""
        
        # Log Tools if present
        if self.tools:
            formatted += "\n  AVAILABLE TOOLS:"
            for tool in self.tools:
                func = tool.get("function", {})
                name = func.get("name", "unknown")
                desc = func.get("description", "no description")
                formatted += f"\n    - {name}: {desc}"
        
        for msg in self.messages:
            # Header
            formatted += f"\n  {msg.role.upper()}: "
            
            # Content handling
            if isinstance(msg.content, str):
                content_lines = msg.content.splitlines()
                if content_lines:
                    formatted += content_lines[0]
                    for line in content_lines[1:]:
                        formatted += f"\n    {line}"
                else:
                    formatted += "(empty)"
            else:
                # Multimodal or complex content
                formatted += json.dumps(msg.content, indent=2).replace("\n", "\n    ")

            # Metadata (Tool calls/ID)
            if msg.name:
                formatted += f" [name={msg.name}]"
            if msg.tool_call_id:
                formatted += f" [id={msg.tool_call_id}]"
            
            if msg.tool_calls:
                formatted += "\n    TOOL_CALLS: " + json.dumps(msg.tool_calls, indent=2).replace("\n", "\n    ")
                
        return formatted
