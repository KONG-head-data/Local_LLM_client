from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = None # Support for {"type": "json_object"}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = None
