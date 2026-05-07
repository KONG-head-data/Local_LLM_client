from vllm import AsyncLLMEngine, AsyncEngineArgs
from .config import MODEL_PATH

# Initialize vLLM Engine (lazy load if possible or handle error)
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        if not MODEL_PATH:
            raise RuntimeError("MODEL_PATH environment variable must be set to the GGUF file path.")
        
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            tokenizer=None, 
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
        )
        _engine = AsyncLLMEngine.from_engine_args(engine_args)
    return _engine
