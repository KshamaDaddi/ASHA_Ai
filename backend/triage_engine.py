def analyze_symptoms(symptoms):

    symptoms = symptoms.lower()

    risk_score = 0
    detected = []

    emergency_keywords = {
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
        "dizziness": 2
    }

    for keyword, score in emergency_keywords.items():

        if keyword in symptoms:

            risk_score += score
            detected.append(keyword)

    # Risk classification
    if risk_score >= 8:
        level = "🔴 HIGH EMERGENCY"

    elif risk_score >= 4:
        level = "🟡 MODERATE RISK"

    else:
        level = "🟢 LOW RISK"

    # Recommendation
    if level == "🔴 HIGH EMERGENCY":
        recommendation = (
            "Seek immediate medical attention."
        )

    elif level == "🟡 MODERATE RISK":
        recommendation = (
            "Monitor symptoms and consult doctor soon."
        )

    else:
        recommendation = (
            "Basic precautions and rest recommended."
        )

    return {
        "risk_level": level,
        "detected_symptoms": detected,
        "risk_score": risk_score,
        "recommendation": recommendation
    }