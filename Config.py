# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASHA-AI  |  config.py
# Central configuration: prompts, language maps, raw Q&A data, triage keywords
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME     = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
ADAPTER_DIR    = "asha_ai_adapter"
MAX_SEQ_LEN    = 1024

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_EPOCHS        = 3
TRAIN_BATCH_SIZE    = 2
GRAD_ACCUM_STEPS    = 4
LEARNING_RATE       = 2e-4
WARMUP_RATIO        = 0.1
DATA_REPEAT         = 10        # augmentation multiplier for small dataset

# ── RAG ───────────────────────────────────────────────────────────────────────
EMBED_MODEL    = "all-MiniLM-L6-v2"
CHROMA_PATH    = "data/chroma_db"
COLLECTION     = "asha_knowledge"
RAG_TOP_K      = 2

# ── TTS ───────────────────────────────────────────────────────────────────────
WHISPER_MODEL  = "base"          # ~142 MB, runs offline on CPU/GPU
TTS_MAX_CHARS  = 500

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are ASHA-AI, a trusted health assistant for ASHA (Accredited Social Health Activist) "
    "workers in rural India. Provide clear, numbered, step-by-step triage guidance based on "
    "IMNCI protocols and ASHA training manuals. Always prioritise patient safety. "
    "When uncertain, advise referral. Use simple language."
)

# ── Language maps ─────────────────────────────────────────────────────────────
ALL_LANGS = [
    "English", "Kannada", "Hindi", "Telugu", "Tamil",
    "Marathi", "Bengali", "Gujarati", "Malayalam", "Punjabi", "Odia",
]

LANG_GT = {
    "Kannada":   "kn", "Hindi":     "hi", "Telugu":    "te", "Tamil":     "ta",
    "Marathi":   "mr", "Bengali":   "bn", "Gujarati":  "gu", "Malayalam": "ml",
    "Punjabi":   "pa", "Odia":      "or", "Assamese":  "as", "Urdu":      "ur",
}

LANG_WHISPER = {
    **LANG_GT,
    "English": "en",
    "auto":    None,
}

LANG_TESSERACT = {
    "Kannada": "kan", "Hindi": "hin", "Telugu": "tel",
    "Tamil":   "tam", "Malayalam": "mal", "Marathi": "mar",
}

# ── Triage keywords ───────────────────────────────────────────────────────────
STRONG_EMERGENCY_KW = [
    "emergency", "call 108", "call 102", "life-threatening", "refer immediately",
    "without delay", "at once", "critically ill",
]
REFER_KW = [
    "refer", "phc", "chc", "hospital", "nrc", "ambulance",
    "immediate", "do not delay", "urgent",
]
HOME_KW = [
    "can be managed at home", "manage at home", "mild", "home treatment",
    "home management", "monitor for", "return if condition worsens",
]

# ── Raw training Q&A (IMNCI / ASHA protocols) ─────────────────────────────────
RAW_DATA = [
    # Paediatric emergencies
    {
        "q": "A 2-year-old child has had fever for 4 days and is breathing very fast with chest in-drawing. What should I do?",
        "a": (
            "This child has SEVERE PNEUMONIA — a medical emergency.\n"
            "1. Do NOT give home treatment and wait.\n"
            "2. Call 108 (emergency ambulance) immediately.\n"
            "3. If amoxicillin syrup is available, give first dose before leaving.\n"
            "4. Keep child upright to ease breathing.\n"
            "5. Refer to the nearest PHC or CHC without delay."
        ),
    },
    {
        "q": "A 6-month-old baby has convulsions (fits) with high fever. What should I do?",
        "a": (
            "Convulsions with fever = EMERGENCY.\n"
            "1. Lay child on side so they do not choke.\n"
            "2. Do NOT put anything in the mouth.\n"
            "3. Do NOT restrain the child.\n"
            "4. Call 108 immediately.\n"
            "5. After fit stops, wrap child warmly and go to hospital at once."
        ),
    },
    {
        "q": "A child cannot drink or breastfeed and is very sleepy and difficult to wake. What should I do?",
        "a": (
            "These are GENERAL DANGER SIGNS — this child is critically ill.\n"
            "1. Call 108 immediately.\n"
            "2. Do not give food or drink by mouth — risk of choking.\n"
            "3. Keep child warm.\n"
            "4. Note time symptoms started.\n"
            "5. Refer to district hospital with paediatric ward."
        ),
    },
    {
        "q": "A child has diarrhoea and is very thirsty, eyes are sunken, skin pinch goes back slowly. What should I do?",
        "a": (
            "This child has SEVERE DEHYDRATION.\n"
            "1. Start ORS (oral rehydration solution) immediately — small sips every 2 minutes.\n"
            "2. Do NOT give plain water or soft drinks.\n"
            "3. If child cannot drink — call 108 for IV fluids at hospital.\n"
            "4. Refer to PHC now.\n"
            "5. Give zinc tablets: 20mg/day for 14 days after recovery."
        ),
    },
    {
        "q": "A child has diarrhoea but is drinking well and not dehydrated. What should I do?",
        "a": (
            "This looks like mild diarrhoea — can be managed at home.\n"
            "1. Give ORS after each loose stool.\n"
            "2. Continue breastfeeding if baby is feeding.\n"
            "3. Give zinc tablets 10mg/day for 10 days (under 6 months) or 20mg/day (over 6 months).\n"
            "4. Continue normal feeding — do not stop food.\n"
            "5. Return immediately if child stops drinking, has blood in stool, or condition worsens."
        ),
    },
    {
        "q": "A child has mild fever for 1 day, is eating and playing normally, and has no other symptoms. What should I do?",
        "a": (
            "This looks like a mild fever — can be managed at home.\n"
            "1. Give paracetamol (15mg/kg every 6 hours) for fever above 38.5°C.\n"
            "2. Sponge with lukewarm water.\n"
            "3. Give extra fluids — ORS, coconut water, or breast milk.\n"
            "4. Monitor for 2 days.\n"
            "5. Return to ASHA or go to PHC if: fever lasts more than 2 days, child stops eating, or gets worse."
        ),
    },
    {
        "q": "A 3-year-old child's MUAC (mid upper arm circumference) measurement is 10.5 cm. What does this mean and what should I do?",
        "a": (
            "MUAC 10.5 cm = SEVERE ACUTE MALNUTRITION (SAM). Red zone.\n"
            "1. Refer child to Nutritional Rehabilitation Centre (NRC) immediately.\n"
            "2. Do NOT just give advice and send home.\n"
            "3. Check for oedema (swelling) on both feet — press for 3 seconds. If pits = more serious.\n"
            "4. Check for medical complications: fever, fast breathing, not eating.\n"
            "5. Enrol in POSHAN Abhiyan. Counsel mother on therapeutic feeding (RUTF)."
        ),
    },
    {
        "q": "A child's MUAC is 12 cm. What does this mean?",
        "a": (
            "MUAC 12 cm = MODERATE ACUTE MALNUTRITION (MAM). Yellow zone.\n"
            "1. Enrol child in supplementary feeding programme.\n"
            "2. Give IFA (iron-folic acid) syrup daily.\n"
            "3. Counsel mother on nutritious foods: eggs, dal, leafy vegetables, fortified rice.\n"
            "4. Measure MUAC again in 2 weeks.\n"
            "5. Refer to NRC if MUAC falls below 11.5 cm or child develops complications."
        ),
    },
    # Newborn care
    {
        "q": "A newborn baby (3 days old) is not feeding well and crying abnormally. What should I do?",
        "a": (
            "A newborn not feeding = DANGER SIGN. Act immediately.\n"
            "1. Check for fast breathing (above 60 per minute).\n"
            "2. Check temperature — cold baby (below 36.5°C) or hot (above 37.5°C) = danger.\n"
            "3. Check skin colour — yellow or blue = danger.\n"
            "4. ANY danger sign = call 102 (free ambulance for newborns under JSSK).\n"
            "5. Do NOT try home remedies — refer to hospital immediately."
        ),
    },
    {
        "q": "A newborn has yellow skin (jaundice) at 18 hours of age. What should I do?",
        "a": (
            "Jaundice within 24 hours of birth = PATHOLOGICAL JAUNDICE = EMERGENCY.\n"
            "1. Refer to hospital immediately for phototherapy (light treatment).\n"
            "2. Do NOT apply turmeric or home remedies.\n"
            "3. Do NOT delay — untreated early jaundice causes brain damage.\n"
            "4. Call 102 for free ambulance.\n"
            "5. Keep baby warm and breastfeeding during transport."
        ),
    },
    {
        "q": "A newborn baby looks healthy, is feeding well and crying normally. What routine care should I give?",
        "a": (
            "Good signs! Follow routine newborn care:\n"
            "1. Keep baby skin-to-skin with mother for warmth.\n"
            "2. Ensure breastfeeding within 1 hour of birth and every 2 hours.\n"
            "3. Do NOT bathe baby for 48 hours.\n"
            "4. Apply nothing to umbilical cord — keep dry and clean.\n"
            "5. Give Vitamin K injection (if available) and eye drops.\n"
            "6. Schedule home visit on day 3, 7, and 28."
        ),
    },
    # Maternal health
    {
        "q": "A pregnant woman in 8th month has severe headache, swollen hands and face, and blurred vision. What should I do?",
        "a": (
            "These are signs of PRE-ECLAMPSIA — life-threatening for mother and baby.\n"
            "1. Call 102 immediately (free ambulance for pregnant women).\n"
            "2. Check blood pressure if available — 140/90 or higher = danger.\n"
            "3. Do NOT give aspirin or pain relievers.\n"
            "4. Lay her on her left side.\n"
            "5. Refer to FRU (First Referral Unit) or district hospital with SNCU."
        ),
    },
    {
        "q": "A pregnant woman has a small amount of bleeding from vagina at 6 months pregnancy. What should I do?",
        "a": (
            "Any vaginal bleeding in pregnancy = DANGER SIGN.\n"
            "1. Refer to PHC or hospital immediately — do not wait.\n"
            "2. Could be placenta praevia (low-lying placenta) — very dangerous.\n"
            "3. Do NOT examine vaginally yourself.\n"
            "4. Call 102 if bleeding is heavy.\n"
            "5. Note time bleeding started and how much — tell doctor."
        ),
    },
    {
        "q": "A pregnant woman is in her 7th month and has not had any ANC visit yet. What should I do?",
        "a": (
            "She has missed important antenatal care. Register her today.\n"
            "1. Take her to the nearest sub-centre or PHC immediately.\n"
            "2. Ensure she gets: haemoglobin test, blood group, BP check, urine test, weight.\n"
            "3. Give iron-folic acid (IFA) tablets and TT vaccination.\n"
            "4. Register her under JSY/JSSK for free delivery benefits.\n"
            "5. Plan at least 2 more ANC visits before delivery."
        ),
    },
    {
        "q": "A woman who delivered 2 weeks ago is feeling very sad, not eating, and says she does not want to live. What should I do?",
        "a": (
            "This woman may have POSTNATAL DEPRESSION with suicidal thoughts. This is a medical emergency.\n"
            "1. Stay with her — do not leave her alone.\n"
            "2. Listen without judgment. Do NOT dismiss her feelings.\n"
            "3. Remove access to harmful objects if possible.\n"
            "4. Call 108 or take her to district hospital mental health unit today.\n"
            "5. Involve a trusted family member immediately."
        ),
    },
    # Adult emergencies
    {
        "q": "An elderly man has chest pain spreading to his left arm and is sweating a lot. What should I do?",
        "a": (
            "This is a possible HEART ATTACK. This is a medical emergency.\n"
            "1. Call 108 immediately. Do NOT wait to see if it improves.\n"
            "2. Make him sit or lie down — do not let him walk.\n"
            "3. If aspirin is available and he is not allergic — give 300mg to chew.\n"
            "4. Loosen tight clothing. Keep him calm.\n"
            "5. Note exact time symptoms started — the doctor must know this."
        ),
    },
    {
        "q": "An adult woman suddenly has weakness on one side of her body and cannot speak properly. What should I do?",
        "a": (
            "This is a possible STROKE. Every minute counts — brain cells are dying.\n"
            "1. Call 108 immediately.\n"
            "2. Do NOT give food or water — she may choke.\n"
            "3. Lay her on her side (recovery position).\n"
            "4. Note exact time symptoms started — hospital must know for treatment.\n"
            "5. The nearest hospital with a CT scan must be reached within 4.5 hours."
        ),
    },
    {
        "q": "A person has been bitten by a snake. What should I do?",
        "a": (
            "Snake bite = EMERGENCY. Do not waste time.\n"
            "1. Call 108 immediately.\n"
            "2. Keep the person calm and still — movement spreads venom faster.\n"
            "3. Do NOT cut the wound, suck out venom, apply tourniquet, or apply herbs.\n"
            "4. Remove tight clothing and jewellery from affected limb.\n"
            "5. Mark swelling boundary with pen and time. Nearest hospital with anti-venom must be reached fast."
        ),
    },
    # Prevention & immunisation
    {
        "q": "Which vaccines should a 6-week-old baby receive?",
        "a": (
            "At 6 weeks (under the Universal Immunisation Programme):\n"
            "1. DPT-1 (diphtheria, pertussis, tetanus)\n"
            "2. OPV-1 (oral polio)\n"
            "3. Hepatitis B-2\n"
            "4. Hib-1 (Haemophilus influenzae)\n"
            "5. Rotavirus-1\n"
            "6. PCV-1 (pneumococcal)\n"
            "All free at government ANM/sub-centre clinics. Record in the immunisation card."
        ),
    },
    {
        "q": "How can I identify and report a malaria case?",
        "a": (
            "Malaria symptoms: fever with chills and sweating, cyclical fever, headache, vomiting.\n"
            "1. Use RDT (Rapid Diagnostic Test) kit to test — training given to all ASHAs.\n"
            "2. If RDT positive: give ACT (artemisinin combination therapy) as per protocol.\n"
            "3. Report to sub-centre/PHC within 24 hours.\n"
            "4. Advise family to use insecticide-treated bed nets.\n"
            "5. Drain any stagnant water near the house."
        ),
    },
]

# ── ASHA knowledge base (for RAG) ─────────────────────────────────────────────
KNOWLEDGE_BASE = [
    ("ANC Danger Signs",
     "Refer immediately for: severe headache, blurred vision, swelling of face/hands, "
     "vaginal bleeding, severe abdominal pain, fever above 38C, reduced fetal movements, "
     "fits or convulsions, labour before 37 weeks. These may indicate pre-eclampsia, "
     "eclampsia, antepartum haemorrhage, or preterm labour."),

    ("ANC Routine Protocol",
     "Minimum 4 ANC visits: first before 12 weeks, then at 14-26 weeks, 28-34 weeks, 36 weeks. "
     "At each visit: weight, BP, fundal height, fetal heart rate, haemoglobin, urine protein. "
     "IFA tablets from 12 weeks. TT injection at 16 and 20 weeks. "
     "Register under JSY scheme for cash incentives at institutional delivery."),

    ("IMNCI - Pneumonia Classification",
     "Fast breathing thresholds: under 2 months above 60/min, 2-12 months above 50/min, "
     "1-5 years above 40/min. Chest in-drawing = severe pneumonia — refer. "
     "Stridor at rest = very severe — emergency referral. "
     "Non-severe pneumonia: amoxicillin 40mg/kg/day in 2 doses for 5 days, treat at home."),

    ("IMNCI - Dehydration",
     "No dehydration: ORS after each stool, zinc 10-20mg/day for 14 days, continue feeding. "
     "Some dehydration (2 of: restless, sunken eyes, thirsty, slow skin pinch): "
     "ORS 75ml/kg over 4 hours at health facility. "
     "Severe dehydration (lethargic, very sunken eyes, unable to drink, very slow skin pinch): "
     "refer immediately for IV rehydration — this is an emergency."),

    ("Newborn Danger Signs",
     "Refer immediately if: not feeding, fast breathing (above 60/min), low temperature below 35.5C, "
     "high temperature above 37.5C, jaundice in first 24 hours, yellow palms and soles, "
     "bleeding, fits, bulging fontanelle, severe skin pustules. "
     "Call 102 for free ambulance under JSSK scheme."),

    ("Malnutrition - MUAC Classification",
     "MUAC colour zones: Green 12.5cm and above (normal, no action), "
     "Yellow 11.5-12.5cm (MAM - moderate acute malnutrition, supplementary feeding), "
     "Red below 11.5cm (SAM - severe acute malnutrition, NRC referral required). "
     "Check for bilateral pitting oedema — both feet pressed 3 seconds and pits = SAM regardless of MUAC."),

    ("Immunisation Schedule - UIP India",
     "Birth: BCG, OPV-0, Hep B-0. 6 weeks: DPT-1, OPV-1, Hib-1, Rotavirus-1, PCV-1, Hep B-1. "
     "10 weeks: DPT-2, OPV-2. 14 weeks: DPT-3, OPV-3, IPV-1, Hib-3, Rotavirus-3, PCV-3, Hep B-3. "
     "9 months: MR-1, VitA-1. 16-24 months: DPT-B1, OPV-B, MR-2, VitA-2. "
     "All vaccines free at government sub-centres under Universal Immunisation Programme."),

    ("Malaria Management",
     "Symptoms: fever with chills and rigors, sweating, headache, vomiting. "
     "Use ASHA RDT kit to test blood. Pf positive: ACT (artemether-lumefantrine) for 3 days. "
     "Pv positive: chloroquine 3 days plus primaquine 14 days (not in G6PD deficiency or pregnancy). "
     "Report to sub-centre within 24 hours. Advise ITN (insecticide-treated net) use."),

    ("Snake Bite First Aid",
     "Keep patient calm and still — movement spreads venom faster. "
     "Remove tight clothing and jewellery from affected limb. "
     "Mark swelling boundary with pen, note time. "
     "Do NOT: cut wound, suck venom, apply tourniquet, apply herbal remedies. "
     "Call 108 immediately. Nearest hospital with antivenom must be reached fast — "
     "tell them it was a snake bite before arrival."),

    ("Postnatal Depression",
     "Signs: persistent sadness after delivery, not eating, poor sleep, not caring for baby, "
     "feeling hopeless, crying often, thoughts of self-harm. "
     "This is a medical condition, not weakness. "
     "Listen without judgment. Involve family. Refer to PHC or district DMHP programme. "
     "If suicidal — treat as emergency, do not leave alone, call 108."),

    ("Fever in Adults - Malaria vs Dengue vs Typhoid",
     "Malaria: cyclical fever with chills, sweating. Test with RDT. "
     "Dengue: high fever, severe headache, pain behind eyes, joint/muscle pain, rash. "
     "Do NOT give aspirin for dengue — risk of bleeding. Give paracetamol only. "
     "Refer if: bleeding from any site, severe abdominal pain, persistent vomiting, lethargy. "
     "Typhoid: prolonged fever, abdominal pain, constipation. Refer for blood test."),

    ("JSY JSSK Schemes",
     "JSY (Janani Suraksha Yojana): cash incentive for institutional delivery. "
     "Rural BPL: Rs 1400, Urban BPL: Rs 1000. ASHA gets Rs 600 per case in LPS states. "
     "JSSK (Janani Shishu Suraksha Karyakram): completely free services for pregnant women "
     "and sick newborns at government facilities — free drugs, diagnostics, blood, diet, transport. "
     "102 ambulance for delivery transport is free. 108 for obstetric emergencies."),
]
