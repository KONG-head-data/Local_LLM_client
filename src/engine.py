from llama_cpp import Llama
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
        # Note: Reverted Q8 KV cache to default F16 due to CUDA SET_ROWS compatibility with Gemma 2
        _engine = Llama(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_gpu_layers=-1,
            flash_attn=True,
            n_batch=512,           # Increased batch size for faster prompt processing
            n_ubatch=512,          # Micro-batch size
            n_threads=10,           # Set to half of physical cores for optimal background tasks,
            verbose=False,
        )
        logger.info("Model loaded successfully with throughput optimizations.")
    return _engine
