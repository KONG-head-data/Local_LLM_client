import os
import sys
from pathlib import Path
from loguru import logger

# Configure Loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

# Configuration via environment variables
MODEL_PATH = os.getenv("MODEL_PATH")

# Default model discovery
if not MODEL_PATH:
    # Try to find a default model in the models/Gemma directory
    project_root = Path(__file__).parent.parent
    default_model_dir = project_root / "models" / "Gemma"
    
    # Prefer Q4 version for speed/balance, fallback to any .gguf
    default_model = default_model_dir / "gemma-4-E4B-it-UD-Q4_K_XL.gguf"
    
    if default_model.exists():
        MODEL_PATH = str(default_model)
        logger.info(f"Using default model: {MODEL_PATH}")
    else:
        logger.warning("MODEL_PATH not set and no default model found in models/Gemma/")
