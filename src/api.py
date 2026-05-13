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

def format_response_for_log(content: str) -> Dict[str, Any]:
    """
    Parses a raw Gemma 4 response string into a clean JSON structure for logging.
    """
    # 1. Extract Thinking
    thinking = ""
    channel_match = re.search(r'<\|channel>thought\n(.*?)<channel|>', content, re.DOTALL)
    if channel_match:
        thinking = channel_match.group(1).strip()
    
    # 2. Extract Tool Calls
    tool_calls = []
    tc_matches = re.findall(r'<\|tool_call\|?>(.*?)<tool_call\|>', content, re.DOTALL)
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

    # 3. Extract Final Content
    clean_content = content
    clean_content = re.sub(r'<\|channel>.*?<channel|>', '', clean_content, flags=re.DOTALL)
    clean_content = re.sub(r'<\|tool_call\|?>(.*?)<tool_call\|>', '', clean_content, flags=re.DOTALL).strip()

    return {
        "thinking": thinking if thinking else None,
        "tool_calls": tool_calls if tool_calls else None,
        "content": clean_content if clean_content else None
    }

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
        chat_template = get_chat_template()
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

    async def run_agentic_inference(msgs):
        current_messages = list(msgs)
        max_iterations = 5
        total_tokens = 0
        
        for i in range(max_iterations):
            def inference_step():
                if chat_template:
                    env = jinja2.Environment(loader=jinja2.BaseLoader())
                    template = env.from_string(chat_template)
                    bos_token = "<bos>"
                    try: bos_token = llm.token_get_text(llm.token_bos()).decode('utf-8')
                    except: pass

                    prompt = template.render(
                        messages=current_messages,
                        bos_token=bos_token,
                        add_generation_prompt=True,
                        enable_thinking=request.thinking,
                        tools=request.tools or []
                    )
                    
                    if prompt.startswith("<bos>"):
                        prompt = prompt.replace("<bos>", "", 1)
                    
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
            choice = response['choices'][0]
            content = choice['message'].get('content', '') if 'message' in choice else choice.get('text', '')
            
            tool_calls = re.findall(r'<\|tool_call\|>(.*?)<tool_call\|>', content, re.DOTALL)
            
            if not tool_calls:
                return response, current_messages, total_tokens
            
            logger.info(f"Agent Iteration {i+1}: Model requested {len(tool_calls)} tool(s).")
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
        async def async_chat_stream():
            def run_chat_inference():
                if chat_template:
                    env = jinja2.Environment(loader=jinja2.BaseLoader())
                    template = env.from_string(chat_template)
                    bos_token = "<bos>"
                    try: bos_token = llm.token_get_text(llm.token_bos()).decode('utf-8')
                    except: pass
                    prompt = template.render(messages=messages, bos_token=bos_token, add_generation_prompt=True, enable_thinking=request.thinking, tools=[])
                    
                    if prompt.startswith("<bos>"):
                        prompt = prompt.replace("<bos>", "", 1)

                    return llm.create_completion(
                        prompt=prompt, 
                        max_tokens=request.max_tokens, 
                        temperature=request.temperature, 
                        top_p=request.top_p, 
                        stream=True, 
                        stop=["<end_of_turn>", "<turn|>"]
                    )
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
            
            duration = time.perf_counter() - start_time
            throughput = token_count / duration if duration > 0 else 0
            logger.info(f"Chat stream finished. Throughput: {throughput:.2f} tokens/sec (Total: {token_count} tokens in {duration:.2f}s)")

        return StreamingResponse(async_chat_stream(), media_type="text/plain")

    response, final_history, total_tokens = await run_agentic_inference(messages)
    duration = time.perf_counter() - start_time
    choice = response['choices'][0]
    content = choice['message'].get('content', '') if 'message' in choice else choice.get('text', '')
    
    # Log the response in a clean, parsed JSON format
    log_data = {"role": "assistant"}
    log_data.update(format_response_for_log(content))
    logger.info(f"Chat Response: {json.dumps(log_data, indent=2)}")

    throughput = total_tokens / duration if duration > 0 else 0
    logger.info(f"Chat agent finished. Throughput: {throughput:.2f} tokens/sec (Total: {total_tokens} tokens in {duration:.2f}s)")

    return {
        "message": {"role": "assistant", "content": content},
        "usage": response['usage'],
        "history": final_history if len(final_history) > len(messages) else None
    }


@router.get("/health")
async def health():
    return {"status": "ok"}
