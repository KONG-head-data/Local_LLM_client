import json
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pathlib import Path

from .schemas import GenerateRequest, ChatRequest
from .config import logger, MODEL_TYPE
from .services import ServiceFactory

router = APIRouter()

# Initialize service based on model type
project_root = str(Path(__file__).parent.parent)
service = ServiceFactory.get_service(MODEL_TYPE, project_root)

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Chat Request: {request.format_for_log()}")
        result = await service.chat(request)
        return result
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    return {"status": "ok"}
