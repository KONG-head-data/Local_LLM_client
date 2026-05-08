import os
from pathlib import Path

# Configuration via environment variables
MODEL_PATH = os.getenv("MODEL_PATH")

# Default model discovery
if not MODEL_PATH:
    # Try to find a default model in the models/Gemma directory
    project_root = Path(__file__).parent.parent
    default_model_dir = project_root / "models" / "Gemma"
    
    # Prefer Q4 version for speed/balance, fallback to any .gguf
    default_model = default_model_dir / "gemma-4-E4B-it-Q4_K_M.gguf"
    
    if default_model.exists():
        MODEL_PATH = str(default_model)
        print(f"Using default model: {MODEL_PATH}")
    else:
        print("WARNING: MODEL_PATH not set and no default model found in models/Gemma/")
