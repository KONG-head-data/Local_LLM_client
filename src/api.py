import time
from typing import AsyncGenerator, Optional
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

    if request.stream:
        async def async_stream_results():
            token_count = 0
            # Use anyio.to_thread to run the blocking generator
            gen = await anyio.to_thread.run_sync(
                lambda: llm.create_completion(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stream=True
                )
            )
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
    def run_inference():
        return llm.create_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False
        )

    response = await anyio.to_thread.run_sync(run_inference)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    token_count = response['usage']['completion_tokens']
    throughput = token_count / duration if duration > 0 else 0
    
    logger.info(f"Generation finished. Throughput: {throughput:.2f} tokens/sec ({token_count} tokens in {duration:.2f}s)")
    
    return {"text": response['choices'][0]['text']}


@router.post("/chat")
async def chat(request: ChatRequest):
    # Convert messages to llama-cpp format or use a simple template
    prompt = ""
    for msg in request.messages:
        prompt += f"{msg.role.capitalize()}: {msg.content}\n"
    prompt += "Assistant: "

    gen_request = GenerateRequest(
        prompt=prompt,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream
    )
    return await generate(gen_request)


@router.get("/health")
async def health():
    return {"status": "ok"}
