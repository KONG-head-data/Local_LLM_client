import time
import json
from typing import AsyncGenerator, Optional, List, Dict, Any
import anyio
import jinja2
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .engine import get_engine, get_chat_template
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

    def run_inference():
        prompt = request.prompt
        if request.thinking and not prompt.startswith("<|think|>"):
            prompt = f"<|think|>{prompt}"
        
        # Determine if we should use grammar for JSON response format
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
            grammar=grammar
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
        chat_template = get_chat_template()
    except Exception as e:
        logger.error(f"Failed to get engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    start_time = time.perf_counter()

    # Convert Pydantic messages to list of dicts
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    logger.info(f"Chat Request (thinking={request.thinking}): {request.format_for_log()}")

    def run_chat_inference():
        # If we have a custom jinja template, render it manually
        if chat_template:
            try:
                env = jinja2.Environment(loader=jinja2.BaseLoader())
                template = env.from_string(chat_template)
                
                # Get BOS token
                bos_token = ""
                try:
                    bos_token = llm.token_get_text(llm.token_bos()).decode('utf-8')
                except:
                    pass

                prompt = template.render(
                    messages=messages,
                    bos_token=bos_token,
                    add_generation_prompt=True,
                    enable_thinking=request.thinking,
                    tools=[]
                )
                
                # Manual grammar for response_format
                grammar = None
                if request.response_format and request.response_format.get("type") == "json_object":
                    try:
                        from llama_cpp.llama_chat_format import _grammar_for_response_format
                        grammar = _grammar_for_response_format(request.response_format)
                    except Exception as e:
                        logger.warning(f"Failed to create grammar for response_format: {e}")

                # Use base completion with the rendered prompt
                return llm.create_completion(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stream=request.stream,
                    grammar=grammar,
                    stop=["<end_of_turn>", "<turn|>"] 
                )
            except Exception as e:
                logger.error(f"Failed to render Jinja template or run completion: {e}")
                # Fallback handled below
        
        # Default chat completion if no template or rendering failed
        # Note: llama-cpp-python's create_chat_completion internally handles response_format
        return llm.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            response_format=request.response_format
        )

    if request.stream:
        async def async_chat_stream():
            token_count = 0
            gen = await anyio.to_thread.run_sync(run_chat_inference)
            
            # Use a simple flag to detect the type of chunk returned
            for chunk in gen:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    content = ""
                    
                    if 'delta' in choice: # Chat completion chunk
                        content = choice['delta'].get('content', '')
                    elif 'text' in choice: # Basic completion chunk
                        content = choice.get('text', '')
                    
                    if content:
                        token_count += 1
                        yield content
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            throughput = token_count / duration if duration > 0 else 0
            logger.info(f"Chat stream finished. Throughput: {throughput:.2f} tokens/sec ({token_count} tokens in {duration:.2f}s)")

        return StreamingResponse(async_chat_stream(), media_type="text/plain")

    # Non-streaming
    response = await anyio.to_thread.run_sync(run_chat_inference)
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # Handle both completion and chat response formats
    choice = response['choices'][0]
    if 'message' in choice: # Chat response
        message = choice['message']
    else: # Completion response
        message = {"role": "assistant", "content": choice.get('text', '')}
    
    usage = response['usage']
    token_count = usage['completion_tokens']
    throughput = token_count / duration if duration > 0 else 0
    
    logger.info(f"Chat generation finished. Throughput: {throughput:.2f} tokens/sec ({token_count} tokens in {duration:.2f}s)")
    
    return {
        "message": message,
        "usage": usage
    }


@router.get("/health")
async def health():
    return {"status": "ok"}
