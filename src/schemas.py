import json
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    stream: bool = False
    thinking: bool = False
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
    thinking: bool = False
    response_format: Optional[Dict[str, Any]] = None

    def format_for_log(self) -> str:
        formatted = ""
        for msg in self.messages:
            content = msg.content
            
            # Find and pretty-print JSON blocks
            result_parts = []
            pos = 0
            
            while pos < len(content):
                start = content.find('{', pos)
                if start == -1:
                    result_parts.append(content[pos:])
                    break
                
                # Append text before JSON
                result_parts.append(content[pos:start])
                
                # Try to find matching brace
                brace_count = 0
                end = -1
                for i in range(start, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                
                if end != -1:
                    candidate = content[start:end]
                    try:
                        data = json.loads(candidate)
                        pretty = json.dumps(data, indent=2)
                        result_parts.append("\n" + pretty + "\n")
                        pos = end
                    except json.JSONDecodeError:
                        # Not JSON, append the '{' and move on
                        result_parts.append('{')
                        pos = start + 1
                else:
                    # No matching '}', append the rest
                    result_parts.append(content[start:])
                    break
            
            final_content = "".join(result_parts)
            # Collapse multiple newlines into at most two
            final_content = re.sub(r'\n{3,}', '\n\n', final_content)
            
            lines = final_content.splitlines()
            if not lines:
                formatted += f"\n  {msg.role.upper()}: "
            else:
                # Filter out empty lines at the very beginning
                start_idx = 0
                while start_idx < len(lines) and not lines[start_idx].strip():
                    start_idx += 1
                
                if start_idx < len(lines):
                    formatted += f"\n  {msg.role.upper()}: {lines[start_idx]}"
                    for line in lines[start_idx+1:]:
                        formatted += f"\n    {line}"
                else:
                    formatted += f"\n  {msg.role.upper()}: "
        return formatted
