from fastapi import FastAPI
from app.routes.chat import router as chat_router

app = FastAPI(
    title="ASHA AI Offline"
)

app.include_router(chat_router)

@app.get("/")
def home():
    return {
        "message": "ASHA AI Backend Running"
    }