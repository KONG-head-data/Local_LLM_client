import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from vllm import SamplingParams, RequestOutput

from .engine import get_engine
from .schemas import GenerateRequest, ChatRequest

router = APIRouter()

@router.post("/generate")
async def generate(request: GenerateRequest):
    try:
        llm_engine = get_engine()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    request_id = str(uuid.uuid4())
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )

    results_generator = llm_engine.generate(request.prompt, sampling_params, request_id)

    async def stream_results() -> AsyncGenerator[str, None]:
        async for request_output in results_generator:
            yield request_output.outputs[0].text

    if request.stream:
        return StreamingResponse(stream_results(), media_type="text/plain")

    # Non-streaming: wait for the last output
    final_output: Optional[RequestOutput] = None
    async for request_output in results_generator:
        final_output = request_output
    
    if final_output is None:
        raise HTTPException(status_code=500, detail="Inference failed to produce output.")
    
    return {"text": final_output.outputs[0].text}


@router.post("/chat")
async def chat(request: ChatRequest):
    # Simple chat template conversion
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
