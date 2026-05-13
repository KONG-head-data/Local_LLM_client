from pathlib import Path
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
        _engine = Llama(
            model_path=MODEL_PATH,
            n_ctx=32768,
            n_gpu_layers=-1,
            flash_attn=True,
            n_batch=512,
            n_ubatch=512,
            n_threads=10,
            verbose=False,
        )
        logger.info("Model loaded successfully with throughput optimizations.")
    return _engine
