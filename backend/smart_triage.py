def smart_triage(symptoms, age):

    symptoms = symptoms.lower()

    age = int(age) if age else 0

    risk_score = 0

    detected = []

    # =====================================================
    # SYMPTOM RULES
    # =====================================================

    symptom_weights = {

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
        "weakness": 2
    }

    for symptom, score in symptom_weights.items():

        if symptom in symptoms:

            risk_score += score

            detected.append(symptom)

    # =====================================================
    # AGE-BASED RISK
    # =====================================================

    if age >= 60:

        risk_score += 2

    if age <= 5 and (
        "fever" in symptoms or
        "dehydration" in symptoms
    ):

        risk_score += 3

    # =====================================================
    # COMBINATION RULES
    # =====================================================

    if (
        "chest pain" in symptoms and
        "breathing difficulty" in symptoms
    ):

        risk_score += 5

    if (
        "high fever" in symptoms and
        "dehydration" in symptoms
    ):

        risk_score += 4

    if (
        "vomiting" in symptoms and
        "weakness" in symptoms
    ):

        risk_score += 2

    # =====================================================
    # FINAL CLASSIFICATION
    # =====================================================

    if risk_score >= 10:

        level = "🔴 CRITICAL EMERGENCY"

    elif risk_score >= 6:

        level = "🟠 HIGH RISK"

    elif risk_score >= 3:

        level = "🟡 MODERATE RISK"

    else:

        level = "🟢 LOW RISK"

    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    if "CRITICAL" in level:

        recommendation = (
            "Immediate hospitalization required."
        )

    elif "HIGH" in level:

        recommendation = (
            "Consult doctor urgently."
        )

    elif "MODERATE" in level:

        recommendation = (
            "Monitor closely and seek medical advice."
        )

    else:

        recommendation = (
            "Rest and basic precautions recommended."
        )

    return {

        "risk_level": level,

        "risk_score": risk_score,

        "detected_symptoms": detected,

        "recommendation": recommendation
    }