from pathlib import Path
from llama_cpp import Llama
from .config import MODEL_PATH, logger

# Initialize Llama Engine (lazy load)
_engine = None
_chat_template = None

def get_engine():
    global _engine, _chat_template
    if _engine is None:
        if not MODEL_PATH:
            logger.error("MODEL_PATH environment variable must be set to the GGUF file path.")
            raise RuntimeError("MODEL_PATH environment variable must be set to the GGUF file path.")

        logger.info(f"Loading model from: {MODEL_PATH}")
        
        # Load local Jinja chat template if available
        project_root = Path(__file__).parent.parent
        template_path = project_root / "models" / "Gemma" / "chat_template.jinja"
        
        if template_path.exists():
            try:
                with open(template_path, "r") as f:
                    _chat_template = f.read()
                logger.info(f"Loaded custom chat template from: {template_path}")
            except Exception as e:
                logger.warning(f"Failed to load chat template: {e}")

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

def get_chat_template():
    if _engine is None:
        get_engine()
    return _chat_template
