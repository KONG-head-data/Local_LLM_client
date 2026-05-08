from contextlib import asynccontextmanager
from fastapi import FastAPI
from .api import router
from .engine import get_engine
from .config import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Force the system to fully load the model first
    try:
        logger.info("Pre-loading model during startup... this may take a moment.")
        get_engine()
        logger.info("Model fully loaded and ready for inference.")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # We continue so the server can at least start, but endpoints will fail
    yield

app = FastAPI(title="Gemma 4 Local API", lifespan=lifespan)

# Include the API router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    # Using 8000 as the standard port
    uvicorn.run("src.main:app", host="0.0.0.0", port=8888, reload=True)
