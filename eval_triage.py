"""
eval_triage.py
==============
Evaluates the fine-tuned ASHA-AI model on clinical triage decisions.
Computes precision, recall, and F1 for "should refer" vs "home treatment" classification.

This is your hackathon writeup's key benchmark table.

Run after fine-tuning is complete.
"""

import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, classification_report

# ─── Gold-standard eval set ───────────────────────────────────────────────────
# Labels: "REFER" = must refer to facility, "HOME" = can manage at home
EVAL_SET = [
    {"query": "Child has fast breathing and chest in-drawing",          "gold": "REFER"},
    {"query": "Child has mild cold and runny nose, no fever",           "gold": "HOME"},
    {"query": "Pregnant woman has fits (convulsions)",                  "gold": "REFER"},
    {"query": "Newborn not feeding for 12 hours",                       "gold": "REFER"},
    {"query": "Adult has mild diarrhoea, still drinking well",          "gold": "HOME"},
    {"query": "Child MUAC is 10.5 cm with swollen feet",               "gold": "REFER"},
    {"query": "Woman in labour for 24 hours, baby not delivered",       "gold": "REFER"},
    {"query": "Child has fever for 1 day, eating and playing normally", "gold": "HOME"},
    {"query": "Man has chest pain spreading to left arm and sweating",  "gold": "REFER"},
    {"query": "Child has mild fever, no danger signs",                  "gold": "HOME"},
    {"query": "Newborn has yellow skin at 18 hours of age",             "gold": "REFER"},
    {"query": "Pregnant woman has headache and swollen hands",          "gold": "REFER"},
    {"query": "Child has mild diarrhoea, no dehydration signs",         "gold": "HOME"},
    {"query": "Elderly woman has sudden weakness on one side of body",  "gold": "REFER"},
    {"query": "Woman has slight nausea in first trimester",             "gold": "HOME"},
]

REFER_KEYWORDS = [
    "refer", "emergency", "call 108", "call 102", "hospital", "phc", "chc", "clinic",
    "immediate", "do not delay", "urgent", "life-threatening", "nrc",
    "ತಕ್ಷಣ", "ತುರ್ತು", "ರೆಫರ್",  # Kannada
]

def predict_label(response: str) -> str:
    """Simple keyword-based classifier on model output."""
    return "REFER" if any(k in response.lower() for k in REFER_KEYWORDS) else "HOME"


def run_evaluation(model_responses: list[str]) -> dict:
    """
    model_responses: list of strings, one per EVAL_SET item.
    Returns a dict with precision, recall, F1 for REFER class.
    """
    gold_labels = [e["gold"] for e in EVAL_SET]
    pred_labels = [predict_label(r) for r in model_responses]

    label_map = {"REFER": 1, "HOME": 0}
    y_true = [label_map[l] for l in gold_labels]
    y_pred = [label_map[l] for l in pred_labels]

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary")

    print("\n=== ASHA-AI Triage Evaluation ===")
    print(classification_report(y_true, y_pred, target_names=["HOME", "REFER"]))
    print(f"REFER class — Precision: {p:.2%}  Recall: {r:.2%}  F1: {f1:.2%}")
    print("\nIn clinical triage, RECALL is the critical metric.")
    print(f"Missing a REFER case = false negative = potential harm to patient.")
    print(f"Your model recall for REFER class: {r:.2%}")

    # Save results
    results = {
        "precision_refer": round(p, 4),
        "recall_refer": round(r, 4),
        "f1_refer": round(f1, 4),
        "per_example": [
            {
                "query": EVAL_SET[i]["query"],
                "gold": gold_labels[i],
                "predicted": pred_labels[i],
                "correct": gold_labels[i] == pred_labels[i],
                "model_response_snippet": model_responses[i][:120],
            }
            for i in range(len(EVAL_SET))
        ],
    }
    out = Path("data/eval_results.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nFull results saved → {out}")
    return results


# ─── Run against Ollama (local Gemma 4) ───────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import httpx

    OLLAMA_URL   = "http://localhost:11434"
    OLLAMA_MODEL = "gemma4:4b"   # change to your fine-tuned model name

    SYSTEM = (
        "You are ASHA-AI, a triage assistant for rural Indian health workers. "
        "Give clear, step-by-step guidance. Always note if referral is needed."
    )

    async def get_response(query: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user",   "content": query},
                    ],
                    "options": {"temperature": 0.1, "num_predict": 256},
                },
            )
            return r.json()["message"]["content"]

    async def main():
        print(f"Evaluating {len(EVAL_SET)} triage scenarios against {OLLAMA_MODEL}...")
        responses = []
        for i, item in enumerate(EVAL_SET):
            print(f"  [{i+1}/{len(EVAL_SET)}] {item['query'][:60]}...")
            resp = await get_response(item["query"])
            responses.append(resp)
        run_evaluation(responses)

    asyncio.run(main())
