import os

# Configuration via environment variables
MODEL_PATH = os.getenv("MODEL_PATH")

if not MODEL_PATH:
    # We won't raise error here to allow linting/basic startup, 
    # but the engine will fail if it's not set.
    print("WARNING: MODEL_PATH environment variable not set.")
