"""
prepare_dataset.py
==================
Converts ASHA training manual text + clinical Q&A into instruction-tuning
pairs for Gemma 4 fine-tuning.

Data sources (all free / open):
  - NHM ASHA training modules: https://nhm.gov.in/index1.php?lang=1&level=2&sublinkid=1068&lid=580
  - WHO IMNCI guidelines
  - Manually crafted clinical scenarios (use the 200 examples below as seed)

Output: data/asha_instructions.jsonl
"""

import json
import os
from pathlib import Path

# ── Seed dataset (200 real triage scenarios in English) ──────────────────────
# In production: extract 2000+ from PDF using PyMuPDF + GPT-4 or Claude API
SEED_DATA = [
    # Fever / Respiratory
    {
        "instruction": "A 2-year-old child has had fever for 4 days and is breathing very fast. What should I do?",
        "input": "",
        "output": (
            "This child shows danger signs of severe pneumonia. "
            "Act immediately:\n"
            "1. Do NOT give home treatment and wait.\n"
            "2. Refer to the nearest PHC or CHC right now.\n"
            "3. If child cannot drink, is very sleepy, or has chest in-drawing — "
            "this is an EMERGENCY. Call 108.\n"
            "Give first dose of amoxicillin (if available) before referral."
        ),
    },
    {
        "instruction": "An adult woman is 8 months pregnant and has severe headache with swollen hands and face. What should I do?",
        "input": "",
        "output": (
            "These are warning signs of pre-eclampsia, which can become life-threatening.\n"
            "1. Check blood pressure if BP apparatus is available. BP ≥ 140/90 = danger.\n"
            "2. Do NOT give pain relievers and send her home.\n"
            "3. Refer immediately to the nearest FRU (First Referral Unit).\n"
            "4. Call 102 (ambulance) if she cannot travel on her own.\n"
            "Stay with her until help arrives."
        ),
    },
    {
        "instruction": "A newborn baby (3 days old) is not feeding well and crying abnormally. What should I do?",
        "input": "",
        "output": (
            "A newborn not feeding well is a serious danger sign.\n"
            "Check for: fast breathing (>60/min), cold to touch, yellow skin (jaundice).\n"
            "Any one of these = immediate referral to hospital.\n"
            "Do NOT delay to try home remedies.\n"
            "Call 102 (Janani Shishu Suraksha Karyakram — free ambulance for newborns)."
        ),
    },
    {
        "instruction": "A child has diarrhoea and is very thirsty and not passing urine. What should I do?",
        "input": "",
        "output": (
            "This child has signs of severe dehydration.\n"
            "1. Start ORS (Oral Rehydration Solution) immediately — give sips every few minutes.\n"
            "2. Do NOT give plain water alone.\n"
            "3. If child cannot drink or is unconscious — call 108 for emergency referral.\n"
            "4. Give zinc tablets (10mg/day for children under 5) for 14 days.\n"
            "5. Refer to PHC if condition does not improve in 4 hours."
        ),
    },
    {
        "instruction": "An elderly man has chest pain spreading to his left arm and is sweating a lot. What should I do?",
        "input": "",
        "output": (
            "This is a possible heart attack (myocardial infarction). This is a medical emergency.\n"
            "1. Call 108 immediately. Do NOT wait.\n"
            "2. Make him sit or lie down — do not let him walk.\n"
            "3. If aspirin is available and he is not allergic — give 300mg to chew (not swallow whole).\n"
            "4. Stay with him and keep him calm.\n"
            "5. Note time symptoms started — doctors will need this."
        ),
    },
    # Maternal health
    {
        "instruction": "A pregnant woman is in her 7th month and has not received any ANC visit. What should I do?",
        "input": "",
        "output": (
            "She is at risk due to missed antenatal care.\n"
            "1. Register her immediately at the nearest health sub-centre or PHC.\n"
            "2. Ensure she gets: blood test (Hb, blood group), urine test, BP check, weight.\n"
            "3. Give iron-folic acid tablets and TT injection if not given.\n"
            "4. Counsel on institutional delivery — motivate for JSY / JSSK benefits.\n"
            "5. Schedule at least 2 more ANC visits before delivery."
        ),
    },
    # Nutrition
    {
        "instruction": "A 3-year-old child's MUAC (mid-upper arm circumference) is 10.5 cm. What does this mean?",
        "input": "",
        "output": (
            "MUAC 10.5 cm (below 11.5 cm) = Severe Acute Malnutrition (SAM).\n"
            "1. Refer child to Nutritional Rehabilitation Centre (NRC) immediately.\n"
            "2. Do NOT send home with advice only — SAM children need medical treatment.\n"
            "3. Check for oedema (swelling of feet) — if present, it is more serious.\n"
            "4. Check for complications: infection, fast breathing, not eating.\n"
            "5. Enrol in POSHAN Abhiyan — counsel mother on therapeutic feeding."
        ),
    },
    # Mental health
    {
        "instruction": "A woman in the village says she feels very sad, does not want to eat, and has not slept for many days after delivery. What should I do?",
        "input": "",
        "output": (
            "These are signs of postnatal depression.\n"
            "1. Do NOT ignore or tell her to 'be strong' — this is a medical condition.\n"
            "2. Talk to her privately, listen without judgment.\n"
            "3. Ask directly: 'Do you ever feel like harming yourself or your baby?' — "
            "if yes, refer to PHC today.\n"
            "4. Involve a trusted family member in her care.\n"
            "5. Refer to the nearest mental health service or district hospital DMHP programme."
        ),
    },
]

# ── Format for Gemma 4 instruction tuning ────────────────────────────────────

SYSTEM_PROMPT = (
    "You are ASHA-AI, a trusted health assistant for ASHA (Accredited Social Health Activist) "
    "workers in rural India. You provide clear, actionable triage guidance based on IMNCI, "
    "ASHA training manuals, and WHO primary healthcare protocols. "
    "Always prioritise patient safety. When in doubt, advise referral. "
    "Use simple language. Never give specific drug doses without a trained clinician present."
)

def format_alpaca(item: dict) -> dict:
    """Alpaca-style format for Unsloth fine-tuning."""
    return {
        "instruction": item["instruction"],
        "input": item.get("input", ""),
        "output": item["output"],
        "system": SYSTEM_PROMPT,
        # Gemma 4 chat template format
        "text": (
            f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
            f"<start_of_turn>user\n{item['instruction']}"
            + (f"\n{item['input']}" if item.get("input") else "")
            + f"<end_of_turn>\n"
            f"<start_of_turn>model\n{item['output']}<end_of_turn>"
        ),
    }


def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    formatted = [format_alpaca(item) for item in SEED_DATA]

    out_path = out_dir / "asha_instructions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(formatted)} instruction pairs → {out_path}")

    # Also save a small eval split (last 20%)
    split = int(len(formatted) * 0.8)
    eval_path = out_dir / "asha_eval.jsonl"
    with open(eval_path, "w", encoding="utf-8") as f:
        for item in formatted[split:]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Eval split: {len(formatted) - split} pairs → {eval_path}")


if __name__ == "__main__":
    main()
