import time
import json
from typing import AsyncGenerator, Optional, List, Dict, Any
import anyio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .engine import get_engine
from .schemas import GenerateRequest, ChatRequest
from .config import logger

router = APIRouter()

@router.post("/generate")
async def generate(request: GenerateRequest):
    try:
        llm = get_engine()
    except Exception as e:
        logger.error(f"Failed to get engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    start_time = time.perf_counter()

    # Ensure response_format is valid for llama-cpp-python
    response_format = request.response_format
    if response_format is not None and "type" not in response_format:
        response_format = None

    prompt = request.prompt
    if request.thinking:
        if not prompt.startswith("<|think|>"):
            prompt = f"<|think|>{prompt}"
    else:
        if prompt.startswith("<|think|>"):
            prompt = prompt[len("<|think|>"):]

    def run_inference():
        return llm.create_completion(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            response_format=response_format
        )

    if request.stream:
        async def async_stream_results():
            token_count = 0
            gen = await anyio.to_thread.run_sync(run_inference)
            for chunk in gen:
                text = chunk['choices'][0]['text']
                if text:
                    token_count += 1
                    yield text
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            throughput = token_count / duration if duration > 0 else 0
            logger.info(f"Stream finished. Throughput: {throughput:.2f} tokens/sec ({token_count} tokens in {duration:.2f}s)")

        return StreamingResponse(async_stream_results(), media_type="text/plain")

    # Non-streaming
    response = await anyio.to_thread.run_sync(run_inference)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    token_count = response['usage']['completion_tokens']
    throughput = token_count / duration if duration > 0 else 0
    
    logger.info(f"Generation finished. Throughput: {throughput:.2f} tokens/sec ({token_count} tokens in {duration:.2f}s)")
    
    return {"text": response['choices'][0]['text'], "usage": response['usage']}


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        llm = get_engine()
    except Exception as e:
        logger.error(f"Failed to get engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    start_time = time.perf_counter()

    # Ensure response_format is valid for llama-cpp-python
    response_format = request.response_format
    if response_format is not None and "type" not in response_format:
        response_format = None

    # Convert Pydantic messages to list of dicts for llama-cpp-python
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.thinking:
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        if system_msg:
            if not system_msg["content"].startswith("<|think|>"):
                system_msg["content"] = f"<|think|>{system_msg['content']}"
        else:
            messages.insert(0, {"role": "system", "content": "<|think|>"})
    else:
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        if system_msg and system_msg["content"].startswith("<|think|>"):
            system_msg["content"] = system_msg["content"][len("<|think|>"):]

    logger.info(f"Chat Request (thinking={request.thinking}): {request.format_for_log()}")

    def run_chat_inference():
        return llm.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            response_format=response_format
        )

    if request.stream:
        async def async_chat_stream():
            token_count = 0
            gen = await anyio.to_thread.run_sync(run_chat_inference)
            for chunk in gen:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        token_count += 1
                        yield delta['content']
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            throughput = token_count / duration if duration > 0 else 0
            logger.info(f"Chat stream finished. Throughput: {throughput:.2f} tokens/sec ({token_count} tokens in {duration:.2f}s)")

        return StreamingResponse(async_chat_stream(), media_type="text/plain")

    # Non-streaming
    response = await anyio.to_thread.run_sync(run_chat_inference)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    token_count = response['usage']['completion_tokens']
    throughput = token_count / duration if duration > 0 else 0
    
    logger.info(f"Chat generation finished. Throughput: {throughput:.2f} tokens/sec ({token_count} tokens in {duration:.2f}s)")
    logger.info(f"Response content: {json.dumps(response['choices'], indent=2)}")
    
    return {
        "message": response['choices'][0]['message'],
        "usage": response['usage']
    }


@router.get("/health")
async def health():
    return {"status": "ok"}
