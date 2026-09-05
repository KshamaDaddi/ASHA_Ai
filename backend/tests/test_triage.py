from app.services.triage_service import triage


def test_low_risk():
    result = triage("mild cough", 25)
    assert result.risk_level == "LOW RISK"
    assert "cough" in result.detected_symptoms


def test_critical_combination():
    result = triage("chest pain and breathing difficulty", 45)
    assert result.risk_level == "CRITICAL EMERGENCY"
    assert result.risk_score >= 10


def test_negated_symptom_is_not_counted():
    result = triage("I don't have chest pain", 45)
    assert "chest pain" not in result.detected_symptoms


def test_age_risk():
    result = triage("headache", 65)
    assert result.risk_score == 3
    assert result.risk_level == "MODERATE RISK"
