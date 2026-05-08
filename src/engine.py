from llama_cpp import Llama, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K
from .config import MODEL_PATH, logger

# Initialize Llama Engine (lazy load)
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        if not MODEL_PATH:
            logger.error("MODEL_PATH environment variable must be set to the GGUF file path.")
            raise RuntimeError("MODEL_PATH environment variable must be set to the GGUF file path.")
        
        logger.info(f"Loading model from: {MODEL_PATH}")
        # Initialize llama-cpp-python with throughput optimizations
        _engine = Llama(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_gpu_layers=-1,
            flash_attn=True,
            n_batch=512,           # Increased batch size for faster prompt processing
            n_ubatch=512,          # Micro-batch size
            n_threads=6,           # Set to half of physical cores for optimal background tasks
            type_k=GGML_TYPE_Q4_K, # 8-bit KV cache for memory bandwidth efficiency
            type_v=GGML_TYPE_Q4_K,
            verbose=False
        )
        logger.info("Model loaded successfully with throughput optimizations.")
    return _engine
