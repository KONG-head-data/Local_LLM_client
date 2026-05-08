from llama_cpp import Llama
from .config import MODEL_PATH

# Initialize Llama Engine (lazy load)
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        if not MODEL_PATH:
            raise RuntimeError("MODEL_PATH environment variable must be set to the GGUF file path.")
        
        # Initialize llama-cpp-python
        _engine = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,           # Context window
            n_gpu_layers=-1,      # Offload all layers to GPU (-1)
            flash_attn=True,      # Enable Flash Attention
            verbose=False         # Set to True for debugging
        )
    return _engine
