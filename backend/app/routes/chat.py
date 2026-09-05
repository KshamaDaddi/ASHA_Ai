"""Chat API route."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.ollama_service import ask_ai

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    question: str
    response: str


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        response = ask_ai(req.message)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Local AI service is unavailable. Make sure Ollama is running and the configured model is installed.",
        ) from exc

    return ChatResponse(question=req.message, response=response)
