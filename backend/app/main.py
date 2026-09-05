"""ASHA AI FastAPI application."""

from fastapi import FastAPI

from app.routes.chat import router as chat_router
from app.routes.triage import router as triage_router

app = FastAPI(
    title="ASHA AI",
    description="Offline, safety-first healthcare support API.",
    version="1.0.0",
)

app.include_router(chat_router)
app.include_router(triage_router)


@app.get("/", tags=["health"])
def home() -> dict[str, str]:
    return {"message": "ASHA AI Backend Running", "status": "ok"}


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "healthy"}
