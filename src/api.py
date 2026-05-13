import time
import json
import re
from typing import AsyncGenerator, Optional, List, Dict, Any
import anyio
import jinja2
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from pathlib import Path
from .engine import get_engine
from .schemas import GenerateRequest, ChatRequest
from .tools import execute_tool, AVAILABLE_TOOLS
from .config import logger, MODEL_TYPE
from .services import ServiceFactory

router = APIRouter()

# Initialize service based on model type
project_root = str(Path(__file__).parent.parent)
service = ServiceFactory.get_service(MODEL_TYPE, project_root)

def format_response_for_log(content: str) -> Dict[str, Any]:
    """
    Parses a raw response string into a clean JSON structure for logging using the service.
    """
    return service.parse_response(content)

@router.post("/generate")
async def generate(request: GenerateRequest):
    try:
        llm = get_engine()
    except Exception as e:
        logger.error(f"Failed to get engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    start_time = time.perf_counter()

    def run_inference():
        prompt = request.prompt
        # For simple generate, we still do basic formatting if requested
        if request.thinking and not prompt.startswith("<|think|>") and MODEL_TYPE == "gemma":
            prompt = f"<|think|>{prompt}"
        elif request.thinking and not prompt.startswith("<think>") and MODEL_TYPE == "qwen":
            prompt = f"<think>{prompt}"
        
        if prompt.startswith("<bos>"):
            prompt = prompt.replace("<bos>", "", 1)

        grammar = None
        if request.response_format and request.response_format.get("type") == "json_object":
            try:
                from llama_cpp.llama_chat_format import _grammar_for_response_format
                grammar = _grammar_for_response_format(request.response_format)
            except Exception as e:
                logger.warning(f"Failed to create grammar for response_format: {e}")

        return llm.create_completion(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            grammar=grammar,
            stop=service.get_stop_tokens()
        )

    if request.stream:
        async def async_stream_results():
            token_count = 0
            gen = await anyio.to_thread.run_sync(run_inference)
            for chunk in gen:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    text = chunk['choices'][0].get('text', '')
                    if text:
                        token_count += 1
                        yield text
            
            duration = time.perf_counter() - start_time
            throughput = token_count / duration if duration > 0 else 0
            logger.info(f"Stream finished. Throughput: {throughput:.2f} tokens/sec (Total: {token_count} tokens in {duration:.2f}s)")

        return StreamingResponse(async_stream_results(), media_type="text/plain")

    response = await anyio.to_thread.run_sync(run_inference)
    duration = time.perf_counter() - start_time
    text = response['choices'][0]['text']
    token_count = response['usage']['completion_tokens']
    throughput = token_count / duration if duration > 0 else 0
    
    logger.info(f"Generation finished. Throughput: {throughput:.2f} tokens/sec (Total: {token_count} tokens in {duration:.2f}s)")
    logger.info(f"Response Content: {text[:500]}{'...' if len(text) > 500 else ''}")
    
    return {"text": text, "usage": response['usage']}


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        llm = get_engine()
    except Exception as e:
        logger.error(f"Failed to get engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    start_time = time.perf_counter()

    # Convert Pydantic messages to list of dicts
    messages = []
    for m in request.messages:
        msg_dict = {"role": m.role, "content": m.content}
        if m.tool_calls: msg_dict["tool_calls"] = m.tool_calls
        if m.tool_call_id: msg_dict["tool_call_id"] = m.tool_call_id
        if m.name: msg_dict["name"] = m.name
        messages.append(msg_dict)

    logger.info(f"Chat Request (thinking={request.thinking}): {request.format_for_log()}")

    async def run_single_inference(msgs):
        def inference_step():
            prompt = service.render_prompt(
                messages=msgs,
                tools=request.tools or [],
                enable_thinking=request.thinking
            )
            
            grammar = None
            if request.response_format and request.response_format.get("type") == "json_object":
                try:
                    from llama_cpp.llama_chat_format import _grammar_for_response_format
                    grammar = _grammar_for_response_format(request.response_format)
                except: pass

            return llm.create_completion(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                grammar=grammar,
                stop=service.get_stop_tokens()
            )

        response = await anyio.to_thread.run_sync(inference_step)
        content = response['choices'][0].get('text', '')
        # Update response to include message-like structure for consistency
        response['choices'][0]['message'] = {"role": "assistant", "content": content}
        
        return response

    if request.stream and not request.tools:
        # ... (rest of stream logic)
        async def async_chat_stream():
            def run_chat_inference():
                prompt = service.render_prompt(
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
                    stop=service.get_stop_tokens()
                )

            token_count = 0
            gen = await anyio.to_thread.run_sync(run_chat_inference)
            for chunk in gen:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    content = choice.get('text', '')
                    if content:
                        token_count += 1
                        yield content
            
            duration = time.perf_counter() - start_time
            throughput = token_count / duration if duration > 0 else 0
            logger.info(f"Chat stream finished. Throughput: {throughput:.2f} tokens/sec (Total: {token_count} tokens in {duration:.2f}s)")

        return StreamingResponse(async_chat_stream(), media_type="text/plain")

    response = await run_single_inference(messages)
    duration = time.perf_counter() - start_time
    choice = response['choices'][0]
    content = choice['message']['content']
    total_tokens = response['usage']['completion_tokens']
    
    # Parse the response for structured logging and output
    parsed_response = service.parse_response(content)
    
    # Construct a clean log object
    throughput = total_tokens / duration if duration > 0 else 0
    log_data = {
        "metrics": {
            "duration_s": round(duration, 2),
            "throughput_tps": round(throughput, 2),
            "total_tokens": total_tokens
        },
        "usage": response['usage'],
        "response": {
            "role": "assistant",
            "thinking": parsed_response.get("thinking"),
            "tool_calls": parsed_response.get("tool_calls"),
            "content": parsed_response.get("content")
        }
    }
    
    logger.info(f"Completion result:\n{json.dumps(log_data, indent=2)}")

    return {
        "message": {
            "role": "assistant", 
            "content": content,
            "parsed": parsed_response
        },
        "usage": response['usage'],
        "metrics": log_data["metrics"]
    }


@router.get("/health")
async def health():
    return {"status": "ok"}
