import time
import json
import re
from typing import AsyncGenerator, Optional, List, Dict, Any
import anyio
import jinja2
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .engine import get_engine, get_chat_template
from .schemas import GenerateRequest, ChatRequest
from .tools import execute_tool, AVAILABLE_TOOLS
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
            logger.info(f"Stream finished. Throughput: {throughput:.2f} tokens/sec (Total: {token_count} tokens in {duration:.2f}s)")

        return StreamingResponse(async_stream_results(), media_type="text/plain")

    # Non-streaming
    response = await anyio.to_thread.run_sync(run_inference)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
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
        chat_template = get_chat_template()
    except Exception as e:
        logger.error(f"Failed to get engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    start_time = time.perf_counter()

    # Convert Pydantic messages to list of dicts
    messages = []
    for m in request.messages:
        msg_dict = {"role": m.role, "content": m.content}
        if m.tool_calls:
            msg_dict["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            msg_dict["tool_call_id"] = m.tool_call_id
        if m.name:
            msg_dict["name"] = m.name
        messages.append(msg_dict)

    logger.info(f"Chat Request (thinking={request.thinking}): {request.format_for_log()}")

    async def run_agentic_inference(msgs):
        current_messages = list(msgs)
        max_iterations = 5
        total_tokens = 0
        
        for i in range(max_iterations):
            # Step 1: Run LLM
            def inference_step():
                if chat_template:
                    env = jinja2.Environment(loader=jinja2.BaseLoader())
                    template = env.from_string(chat_template)
                    
                    bos_token = ""
                    try:
                        bos_token = llm.token_get_text(llm.token_bos()).decode('utf-8')
                    except:
                        pass

                    prompt = template.render(
                        messages=current_messages,
                        bos_token=bos_token,
                        add_generation_prompt=True,
                        enable_thinking=request.thinking,
                        tools=request.tools or []
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
                        stop=["<end_of_turn>", "<turn|>"] 
                    )
                else:
                    return llm.create_chat_completion(
                        messages=current_messages,
                        tools=request.tools,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        response_format=request.response_format
                    )

            response = await anyio.to_thread.run_sync(inference_step)
            total_tokens += response['usage']['completion_tokens']
            
            # Step 2: Parse response
            choice = response['choices'][0]
            if 'message' in choice:
                content = choice['message'].get('content', '')
            else:
                content = choice.get('text', '')
            
            # Step 3: Check for tool calls in the output text
            tool_calls = re.findall(r'<\|tool_call\|>(.*?)<tool_call\|>', content, re.DOTALL)
            
            if not tool_calls:
                # No more tools, this is the final answer
                return response, current_messages, total_tokens
            
            logger.info(f"Agent Iteration {i+1}: Model requested {len(tool_calls)} tool(s).")
            
            # Step 4: Execute tools and update conversation
            current_messages.append({"role": "assistant", "content": content})
            
            for call_str in tool_calls:
                try:
                    func_name = call_str.split("{")[0].replace("call:", "").strip()
                    tool_result = execute_tool(call_str)
                    tid = f"auto_call_{time.time()}"
                    current_messages.append({
                        "role": "tool",
                        "name": func_name,
                        "tool_call_id": tid,
                        "content": tool_result
                    })
                    logger.info(f"Tool '{func_name}' executed. Result: {tool_result[:100]}...")
                except Exception as e:
                    logger.error(f"Error processing tool call: {e}")
            
        return response, current_messages, total_tokens

    if request.stream and not request.tools:
        # Standard streaming for non-tool requests
        async def async_chat_stream():
            def run_chat_inference():
                if chat_template:
                    env = jinja2.Environment(loader=jinja2.BaseLoader())
                    template = env.from_string(chat_template)
                    bos_token = ""
                    try: bos_token = llm.token_get_text(llm.token_bos()).decode('utf-8')
                    except: pass
                    prompt = template.render(messages=messages, bos_token=bos_token, add_generation_prompt=True, enable_thinking=request.thinking, tools=[])
                    return llm.create_completion(prompt=prompt, max_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p, stream=True, stop=["<end_of_turn>", "<turn|>"])
                return llm.create_chat_completion(messages=messages, max_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p, stream=True)

            token_count = 0
            gen = await anyio.to_thread.run_sync(run_chat_inference)
            for chunk in gen:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    content = choice['delta'].get('content', '') if 'delta' in choice else choice.get('text', '')
                    if content:
                        token_count += 1
                        yield content
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            throughput = token_count / duration if duration > 0 else 0
            logger.info(f"Chat stream finished. Throughput: {throughput:.2f} tokens/sec (Total: {token_count} tokens in {duration:.2f}s)")

        return StreamingResponse(async_chat_stream(), media_type="text/plain")

    # Agentic Loop (Non-streaming or Tool-enabled)
    response, final_history, total_tokens = await run_agentic_inference(messages)
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    choice = response['choices'][0]
    if 'message' in choice:
        message = choice['message']
    else:
        message = {"role": "assistant", "content": choice.get('text', '')}
    
    throughput = total_tokens / duration if duration > 0 else 0
    
    logger.info(f"Chat agent finished. Throughput: {throughput:.2f} tokens/sec (Total: {total_tokens} tokens in {duration:.2f}s)")
    logger.info(f"Chat Response: {json.dumps(message, indent=2)}")
    
    return {
        "message": message,
        "usage": response['usage'],
        "history": final_history if len(final_history) > len(messages) else None
    }


@router.get("/health")
async def health():
    return {"status": "ok"}
