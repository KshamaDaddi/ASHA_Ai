"""Preliminary triage API."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.triage_service import triage

router = APIRouter(prefix="/api/v1", tags=["triage"])


class TriageRequest(BaseModel):
    symptoms: str = Field(..., min_length=1, max_length=4000)
    age: int | None = Field(default=None, ge=0, le=120)


@router.post("/triage")
def run_triage(req: TriageRequest) -> dict[str, object]:
    result = triage(req.symptoms, req.age)
    return result.as_dict()
