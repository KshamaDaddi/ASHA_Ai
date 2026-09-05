"""Deterministic, safety-first preliminary triage engine."""
from dataclasses import dataclass
import re

SYMPTOM_WEIGHTS = {
    "breathing difficulty": 5,
    "shortness of breath": 5,
    "heavy bleeding": 5,
    "chest pain": 5,
    "unconscious": 5,
    "seizure": 5,
    "high fever": 3,
    "dehydration": 3,
    "vomiting": 2,
    "dizziness": 2,
    "weakness": 2,
    "fever": 2,
    "headache": 1,
    "cough": 1,
}

NEGATION_PATTERNS = (
    r"\bno\s+(?:severe\s+)?{symptom}\b",
    r"\bwithout\s+(?:any\s+)?{symptom}\b",
    r"\bdo\s+not\s+have\s+(?:any\s+)?{symptom}\b",
    r"\bdon't\s+have\s+(?:any\s+)?{symptom}\b",
)

@dataclass(frozen=True)
class TriageResult:
    risk_level: str
    risk_score: int
    detected_symptoms: list[str]
    recommendation: str

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())

def contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))

def is_negated(text: str, symptom: str) -> bool:
    escaped = re.escape(symptom)
    return any(re.search(pattern.format(symptom=escaped), text) for pattern in NEGATION_PATTERNS)

def triage(symptoms: str, age: int | None = None) -> TriageResult:
    text = normalize(symptoms)
    patient_age = max(0, int(age or 0))
    score = 0
    detected: list[str] = []
    active: set[str] = set()
    occupied: list[tuple[int, int]] = []

    # Longest-first matching prevents "high fever" + "fever" double counting.
    for symptom, weight in sorted(SYMPTOM_WEIGHTS.items(), key=lambda item: -len(item[0])):
        match = re.search(rf"(?<!\w){re.escape(symptom)}(?!\w)", text)
        if not match or is_negated(text, symptom):
            continue
        start, end = match.span()
        if any(start < old_end and old_start < end for old_start, old_end in occupied):
            continue
        occupied.append((start, end))
        score += weight
        detected.append(symptom)
        active.add(symptom)

    if patient_age >= 60:
        score += 2
    if patient_age <= 5 and {"fever", "high fever", "dehydration"} & active:
        score += 3
    if ({"chest pain", "breathing difficulty"} <= active or
            {"chest pain", "shortness of breath"} <= active):
        score += 5
    if {"high fever", "dehydration"} <= active:
        score += 4
    if {"vomiting", "weakness"} <= active:
        score += 2

    if score >= 10:
        level = "CRITICAL EMERGENCY"
        recommendation = "Seek emergency medical care immediately."
    elif score >= 6:
        level = "HIGH RISK"
        recommendation = "Seek urgent medical evaluation as soon as possible."
    elif score >= 3:
        level = "MODERATE RISK"
        recommendation = "Monitor symptoms closely and seek medical advice."
    else:
        level = "LOW RISK"
        recommendation = "Continue basic precautions and seek medical advice if symptoms worsen."

    return TriageResult(level, score, detected, recommendation)
