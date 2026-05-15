import re
import json
import time
import jinja2
import anyio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from fastapi.responses import StreamingResponse

from .config import logger, MODEL_TYPE
from .engine import get_engine
from .schemas import ChatRequest, ChatResponse, ParsedMessage, ChatMetrics

class BaseModelService(ABC):
    @abstractmethod
    def render_prompt(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, enable_thinking: bool = False) -> str:
        pass

    @abstractmethod
    def parse_response(self, text: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_stop_tokens(self) -> List[str]:
        pass

    @abstractmethod
    async def chat(self, request: ChatRequest) -> Union[ChatResponse, StreamingResponse]:
        pass

class GemmaService(BaseModelService):
    def __init__(self, template_path: str):
        self.template_path = template_path
        with open(template_path, "r") as f:
            self.template_str = f.read()
        self.env = jinja2.Environment(loader=jinja2.BaseLoader())
        self.template = self.env.from_string(self.template_str)

    def render_prompt(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, enable_thinking: bool = False) -> str:
        # Gemma template usually expects model role instead of assistant
        formatted_messages = []
        for msg in messages:
            m = msg.copy()
            if m["role"] == "assistant":
                m["role"] = "model"
            formatted_messages.append(m)

        prompt = self.template.render(
            messages=formatted_messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tools=tools or []
        )
        if prompt.startswith("<bos>"):
            prompt = prompt.replace("<bos>", "", 1)
        return prompt

    def parse_response(self, text: str) -> Dict[str, Any]:
        # Extract Thinking
        thinking = ""
        channel_match = re.search(r'<\|channel>thought\n(.*?)<channel|>', text, re.DOTALL)
        if channel_match:
            thinking = channel_match.group(1).strip()
        
        # Extract Tool Calls
        tool_calls = []
        tc_matches = re.findall(r'<\|tool_call\|?>(.*?)<tool_call\|>', text, re.DOTALL)
        for tc in tc_matches:
            try:
                name = tc.split("{")[0].replace("call:", "").strip()
                args_str = tc[tc.find("{"):]
                args_str = args_str.replace('<|\"|>', '"')
                
                try:
                    json_safe = re.sub(r'(\w+):', r'"\1":', args_str)
                    args = json.loads(json_safe)
                except:
                    args = args_str
                    
                tool_calls.append({"name": name, "arguments": args})
            except:
                tool_calls.append({"raw": tc})

        # Extract Final Content
        clean_content = text
        #clean_content = re.sub(r'<\|channel>.*?<channel\|>', '', clean_content, flags=re.DOTALL)
        #clean_content = re.sub(r'<\|tool_call\|?>(.*?)<tool_call\|>', '', clean_content, flags=re.DOTALL).strip()

        return {
            "thinking": thinking if thinking else None,
            "tool_calls": tool_calls if tool_calls else None,
            "content": clean_content if clean_content else None
        }

    def get_stop_tokens(self) -> List[str]:
        return ["<end_of_turn>", "<turn|>"]

    async def chat(self, request: ChatRequest) -> Union[Dict[str, Any], StreamingResponse]:
        llm = get_engine()
        start_time = time.perf_counter()

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
            
        for m in request.messages:
            msg_dict = {"role": m.role, "content": m.content}
            messages.append(msg_dict)

        async def run_single_inference(msgs):
            def inference_step():
                prompt = self.render_prompt(
                    messages=msgs,
                    tools=request.tools or [],
                    enable_thinking=request.thinking
                )
                
                return llm.create_completion(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=self.get_stop_tokens()
                )

            response = await anyio.to_thread.run_sync(inference_step)
            return response

        if request.stream and not request.tools:
            async def async_chat_stream():
                def run_chat_inference():
                    prompt = self.render_prompt(
                        messages=messages,
                        tools=[],
                        enable_thinking=request.thinking
                    )
                    return llm.create_completion(
                        prompt=prompt, 
                        max_tokens=request.max_tokens, 
                        temperature=request.temperature, 
                        top_p=request.top_p, 
                        stream=True, 
                        stop=self.get_stop_tokens()
                    )

                token_count = 0
                gen = await anyio.to_thread.run_sync(run_chat_inference)
                for chunk in gen:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        content = chunk['choices'][0].get('text', '')
                        if content:
                            token_count += 1
                            yield content
                
                duration = time.perf_counter() - start_time
                throughput = token_count / duration if duration > 0 else 0
                logger.info(f"Gemma chat stream finished. Throughput: {throughput:.2f} t/s")

            return StreamingResponse(async_chat_stream(), media_type="text/plain")

        response = await run_single_inference(messages)
        duration = time.perf_counter() - start_time
        content = response['choices'][0].get('text', '')
        total_tokens = response['usage']['completion_tokens']
        
        parsed_response = self.parse_response(content)
        throughput = total_tokens / duration if duration > 0 else 0
        
        metrics = {
            "duration_s": round(duration, 2),
            "throughput_tps": round(throughput, 2),
            "total_tokens": total_tokens
        }

        logger.info(f"{self.__class__.__name__} chat result:\n{json.dumps({'metrics': metrics,'parsed': parsed_response}, indent=2)}")

        return ChatResponse(
            message={
                "role": "assistant",
                "content": content
            },
            parsed_message=ParsedMessage(**parsed_response),
            usage=response['usage'],
            metrics=ChatMetrics(**metrics)
        )

class QwenService(BaseModelService):
    def __init__(self, template_path: str):
        self.template_path = template_path
        with open(template_path, "r") as f:
            self.template_str = f.read()
        self.env = jinja2.Environment(loader=jinja2.BaseLoader())
        self.template = self.env.from_string(self.template_str)

    def render_prompt(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, enable_thinking: bool = False) -> str:
        prompt = self.template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tools=tools or []
        )
        return prompt

    def parse_response(self, text: str) -> Dict[str, Any]:
        # Extract Thinking
        thinking = ""
        think_match = re.search(r'(.*?)</think>', text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
        
        # Extract Tool Calls
        tool_calls = []
        # Qwen format: <tool_call>\n<function=name>\n<parameter=p1>\nv1\n</parameter>\n</function>\n</tool_call>
        tc_matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        for tc in tc_matches:
            try:
                func_match = re.search(r'<function=(.*?)>', tc)
                if not func_match:
                    continue
                name = func_match.group(1).strip()
                
                args = {}
                param_matches = re.findall(r'<parameter=(.*?)>(.*?)</parameter>', tc, re.DOTALL)
                for p_name, p_val in param_matches:
                    val = p_val.strip()
                    try:
                        args[p_name.strip()] = json.loads(val)
                    except:
                        args[p_name.strip()] = val
                
                tool_calls.append({"name": name, "arguments": args})
            except Exception as e:
                logger.error(f"Failed to parse Qwen tool call: {e}")
                tool_calls.append({"raw": tc})

        # Extract Final Content
        clean_content = text
        #clean_content = re.sub(r'<think>.*?</think>', '', clean_content, flags=re.DOTALL)
        #clean_content = re.sub(r'<tool_call>.*?</tool_call>', '', clean_content, flags=re.DOTALL).strip()

        return {
            "thinking": thinking if thinking else None,
            "tool_calls": tool_calls if tool_calls else None,
            "content": clean_content if clean_content else None
        }

    def get_stop_tokens(self) -> List[str]:
        return ["<|im_end|>", "<|endoftext|>"]

    async def chat(self, request: ChatRequest) -> Union[Dict[str, Any], StreamingResponse]:
        llm = get_engine()
        start_time = time.perf_counter()

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
            
        for m in request.messages:
            msg_dict = {"role": m.role, "content": m.content}
            messages.append(msg_dict)

        async def run_single_inference(msgs):
            def inference_step():
                prompt = self.render_prompt(
                    messages=msgs,
                    tools=request.tools or [],
                    enable_thinking=request.thinking
                )
                
                return llm.create_completion(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=self.get_stop_tokens()
                )

            response = await anyio.to_thread.run_sync(inference_step)
            return response

        if request.stream and not request.tools:
            async def async_chat_stream():
                def run_chat_inference():
                    prompt = self.render_prompt(
                        messages=messages,
                        tools=[],
                        enable_thinking=request.thinking
                    )
                    return llm.create_completion(
                        prompt=prompt, 
                        max_tokens=request.max_tokens, 
                        temperature=request.temperature, 
                        top_p=request.top_p, 
                        stream=True, 
                        stop=self.get_stop_tokens()
                    )

                token_count = 0
                gen = await anyio.to_thread.run_sync(run_chat_inference)
                for chunk in gen:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        content = chunk['choices'][0].get('text', '')
                        if content:
                            token_count += 1
                            yield content
                
                duration = time.perf_counter() - start_time
                throughput = token_count / duration if duration > 0 else 0
                logger.info(f"Qwen chat stream finished. Throughput: {throughput:.2f} t/s")

            return StreamingResponse(async_chat_stream(), media_type="text/plain")

        response = await run_single_inference(messages)
        duration = time.perf_counter() - start_time
        content = response['choices'][0].get('text', '')
        total_tokens = response['usage']['completion_tokens']
        
        parsed_response = self.parse_response(content)
        throughput = total_tokens / duration if duration > 0 else 0
        
        metrics = {
            "duration_s": round(duration, 2),
            "throughput_tps": round(throughput, 2),
            "total_tokens": total_tokens
        }

        logger.info(f"{self.__class__.__name__} chat result:\n{json.dumps({'metrics': metrics,'parsed': parsed_response}, indent=2)}")

        return ChatResponse(
            message={
                "role": "assistant",
                "content": content
            },
            parsed_message=ParsedMessage(**parsed_response),
            usage=response['usage'],
            metrics=ChatMetrics(**metrics)
        )

class ServiceFactory:
    @staticmethod
    def get_service(model_type: str, project_root: str) -> BaseModelService:
        if model_type == "gemma":
            template_path = f"{project_root}/models/Gemma/chat_template.jinja"
            return GemmaService(template_path)
        elif model_type == "qwen":
            # Check for 9B or 4B, default to 9B if both exist or just one
            import os
            qwen_9b = f"{project_root}/models/Qwen-3.5-9B/chat_template.jinja"
            qwen_4b = f"{project_root}/models/Qwen-3.5-4B/chat_template.jinja"
            
            if os.path.exists(qwen_9b):
                return QwenService(qwen_9b)
            elif os.path.exists(qwen_4b):
                return QwenService(qwen_4b)
            else:
                raise FileNotFoundError("Qwen chat template not found")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
