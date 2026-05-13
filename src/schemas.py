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
        def indent(text, spaces):
            return "\n".join(" " * spaces + line for line in text.splitlines())

        # Format Messages
        msg_list = []
        for msg in self.messages:
            content_val = f'"{msg.content}"' if isinstance(msg.content, str) else json.dumps(msg.content, indent=12).strip()
            m = (
                f"ChatMessage(\n"
                f'    role="{msg.role}",\n'
                f'    content={content_val},\n'
                f'    tool_calls={msg.tool_calls},\n'
                f'    tool_call_id={msg.tool_call_id},\n'
                f'    name={msg.name},\n'
                f")"
            )
            msg_list.append(indent(m, 8).strip())
        
        messages_str = "[\n        " + ",\n        ".join(msg_list) + "\n    ]"

        # Format Tools
        if self.tools:
            tools_json = json.dumps(self.tools, indent=4)
            tools_str = indent(tools_json, 4).strip()
        else:
            tools_str = "None"

        # Assemble Final Output
        return (
            f"ChatRequest(\n"
            f"    thinking={self.thinking},\n"
            f"    messages={messages_str},\n"
            f"    tools={tools_str},\n"
            f"    temperature={self.temperature},\n"
            f"    top_p={self.top_p},\n"
            f"    max_tokens={self.max_tokens},\n"
            f"    stream={self.stream},\n"
            f"    response_format={self.response_format},\n"
            f")"
        )
