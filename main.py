import os
import uuid
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, RequestOutput

app = FastAPI(title="vLLM Local GGUF API")

# Configuration via environment variables
MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    # We won't raise error here to allow linting/basic startup, 
    # but the engine will fail if it's not set.
    print("WARNING: MODEL_PATH environment variable not set.")

# Initialize vLLM Engine (lazy load if possible or handle error)
engine = None

def get_engine():
    global engine
    if engine is None:
        if not MODEL_PATH:
            raise RuntimeError("MODEL_PATH environment variable must be set to the GGUF file path.")
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            tokenizer=None, 
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    stream: bool = False


@app.post("/generate")
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


@app.post("/chat")
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


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
