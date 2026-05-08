from typing import AsyncGenerator, Optional
import anyio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .engine import get_engine
from .schemas import GenerateRequest, ChatRequest

router = APIRouter()

@router.post("/generate")
async def generate(request: GenerateRequest):
    try:
        llm = get_engine()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Define the execution in a threadpool to avoid blocking the event loop
    def run_inference():
        return llm.create_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream
        )

    if request.stream:
        async def stream_results() -> AsyncGenerator[str, None]:
            # Run the generator in a threadpool
            def sync_gen():
                for chunk in run_inference():
                    yield chunk['choices'][0]['text']
            
            # Use anyio to iterate over the sync generator in a separate thread
            async for text in anyio.to_thread.run_sync(sync_gen):
                # This is still a bit tricky for real-time streaming from sync gen
                # For now, let's keep it simple or use the native stream if possible
                pass
        
        # Proper async streaming for llama-cpp-python
        async def async_stream_results():
            # Use anyio.to_thread to run the blocking generator
            gen = await anyio.to_thread.run_sync(run_inference)
            for chunk in gen:
                yield chunk['choices'][0]['text']

        return StreamingResponse(async_stream_results(), media_type="text/plain")

    # Non-streaming
    response = await anyio.to_thread.run_sync(run_inference)
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
