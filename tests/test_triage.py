from app.services.triage_engine import triage

def test_emergency_combination():
    result = triage("chest pain and breathing difficulty", 30)
    assert result.risk_level == "CRITICAL EMERGENCY"
    assert result.risk_score >= 10

def test_high_fever_not_double_counted():
    result = triage("high fever", 30)
    assert result.detected_symptoms == ["high fever"]
    assert result.risk_score == 3

def test_negated_symptom_is_not_detected():
    result = triage("I don't have chest pain", 30)
    assert "chest pain" not in result.detected_symptoms
    assert result.risk_score == 0

def test_older_age_adds_risk():
    result = triage("headache", 60)
    assert result.risk_score == 3
    assert result.risk_level == "MODERATE RISK"

def test_young_child_fever_escalates():
    result = triage("fever", 4)
    assert result.risk_score == 5
    assert result.risk_level == "MODERATE RISK"

def test_low_risk_symptom():
    result = triage("cough", 30)
    assert result.risk_score == 1
    assert result.risk_level == "LOW RISK"
