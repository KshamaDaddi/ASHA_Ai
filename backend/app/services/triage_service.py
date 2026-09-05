"""Deterministic safety-first triage service for ASHA AI.

This module intentionally does not use the LLM to make the risk decision.
The rules provide a predictable safety layer; the LLM can explain the result.
"""

from dataclasses import dataclass
from typing import Dict, List
import re


@dataclass(frozen=True)
class TriageResult:
    risk_level: str
    risk_score: int
    detected_symptoms: List[str]
    recommendation: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "detected_symptoms": self.detected_symptoms,
            "recommendation": self.recommendation,
        }


SYMPTOM_WEIGHTS = {
    "chest pain": 5,
    "breathing difficulty": 5,
    "shortness of breath": 5,
    "unconscious": 5,
    "seizure": 5,
    "heavy bleeding": 5,
    "high fever": 3,
    "vomiting": 2,
    "dehydration": 3,
    "cough": 1,
    "fever": 2,
    "headache": 1,
    "dizziness": 2,
    "weakness": 2,
}

NEGATION_PATTERNS = (
    r"\bno\s+(?:severe\s+)?{symptom}\b",
    r"\bwithout\s+(?:any\s+)?{symptom}\b",
    r"\bdo\s+not\s+have\s+(?:any\s+)?{symptom}\b",
    r"\bdon't\s+have\s+(?:any\s+)?{symptom}\b",
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _is_negated(text: str, symptom: str) -> bool:
    return any(re.search(pattern.format(symptom=re.escape(symptom)), text) for pattern in NEGATION_PATTERNS)


def triage(symptoms: str, age: int | None = None) -> TriageResult:
    """Return a deterministic preliminary risk assessment.

    This is a screening aid, not a diagnosis. It deliberately errs toward
    escalation when multiple high-risk signals are present.
    """
    text = _normalise(symptoms or "")
    patient_age = max(0, int(age or 0))

    risk_score = 0
    detected: List[str] = []
    active = set()

    for symptom, weight in SYMPTOM_WEIGHTS.items():
        if symptom in text and not _is_negated(text, symptom):
            risk_score += weight
            detected.append(symptom)
            active.add(symptom)

    if patient_age >= 60:
        risk_score += 2
    if patient_age <= 5 and ({"fever", "high fever", "dehydration"} & active):
        risk_score += 3

    if {"chest pain", "breathing difficulty"} <= active or {
        "chest pain", "shortness of breath"
    } <= active:
        risk_score += 5
    if {"high fever", "dehydration"} <= active:
        risk_score += 4
    if {"vomiting", "weakness"} <= active:
        risk_score += 2

    if risk_score >= 10:
        level = "CRITICAL EMERGENCY"
        recommendation = "Seek emergency medical care immediately."
    elif risk_score >= 6:
        level = "HIGH RISK"
        recommendation = "Seek urgent medical evaluation as soon as possible."
    elif risk_score >= 3:
        level = "MODERATE RISK"
        recommendation = "Monitor symptoms closely and seek medical advice."
    else:
        level = "LOW RISK"
        recommendation = "Continue basic precautions and seek medical advice if symptoms worsen."

    return TriageResult(level, risk_score, detected, recommendation)
