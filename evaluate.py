# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASHA-AI  |  evaluate.py
# Evaluate REFER/HOME classification accuracy on 15 gold-standard triage cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import json
from pathlib import Path

from .config import STRONG_EMERGENCY_KW, REFER_KW, HOME_KW
from .inference import asha_triage

# ── Gold-standard evaluation set ─────────────────────────────────────────────

EVAL_SET = [
    {"query": "Child has fast breathing and chest in-drawing with fever",      "gold": "REFER"},
    {"query": "Child has mild cold and runny nose, no fever, playing well",    "gold": "HOME"},
    {"query": "Pregnant woman has fits and convulsions",                       "gold": "REFER"},
    {"query": "Newborn not feeding for 12 hours",                              "gold": "REFER"},
    {"query": "Adult has mild diarrhoea, still drinking well, no dehydration", "gold": "HOME"},
    {"query": "Child MUAC is 10.5 cm with swollen feet on both sides",         "gold": "REFER"},
    {"query": "Woman in labour for over 24 hours, baby not delivered",         "gold": "REFER"},
    {"query": "Child has fever for 1 day, eating and playing normally",        "gold": "HOME"},
    {"query": "Man has chest pain spreading to left arm and is sweating",      "gold": "REFER"},
    {"query": "Child has mild diarrhoea, no dehydration signs, drinking ORS",  "gold": "HOME"},
    {"query": "Newborn has yellow skin at 18 hours of age",                    "gold": "REFER"},
    {"query": "Pregnant woman has headache and swollen hands at 8 months",     "gold": "REFER"},
    {"query": "Woman has slight nausea in first trimester of pregnancy",        "gold": "HOME"},
    {"query": "Elderly woman has sudden weakness on one side of body",         "gold": "REFER"},
    {"query": "Child has mild runny nose, no danger signs, eating well",       "gold": "HOME"},
]


# ── Binary classifier ─────────────────────────────────────────────────────────

def classify_response(response: str) -> str:
    """
    Classify a triage response as REFER or HOME using keyword matching.

    Priority:
        1. Strong emergency keywords → REFER
        2. Home management keywords  → HOME
        3. Any referral keyword      → REFER
        4. Default                   → HOME
    """
    r = response.lower()
    if any(k in r for k in STRONG_EMERGENCY_KW):
        return "REFER"
    if any(k in r for k in HOME_KW):
        return "HOME"
    if any(k in r for k in REFER_KW):
        return "REFER"
    return "HOME"


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(
    eval_set: list = EVAL_SET,
    output_path: str = "data/eval_results.json",
) -> dict:
    """
    Run triage inference on every case in eval_set, compute metrics, and
    save results to JSON.

    Key metric: REFER recall — a missed referral can cost a patient's life.

    Returns:
        Dict with accuracy, precision, recall, F1, and per-case results.
    """
    from sklearn.metrics import precision_recall_fscore_support, classification_report

    print(f"Running evaluation on {len(eval_set)} triage scenarios…\n")
    results = []

    for item in eval_set:
        response  = asha_triage(item["query"], language="en")
        predicted = classify_response(response)
        correct   = predicted == item["gold"]
        results.append({
            **item,
            "predicted": predicted,
            "correct":   correct,
            "response":  response[:200],
        })
        icon = "✅" if correct else "❌"
        print(f"{icon} [{item['gold']} → {predicted}] {item['query'][:55]}…")

    # ── Metrics ────────────────────────────────────────────────────────────────
    gold_labels = [r["gold"] for r in results]
    pred_labels = [r["predicted"] for r in results]
    label_map   = {"REFER": 1, "HOME": 0}
    y_true      = [label_map[l] for l in gold_labels]
    y_pred      = [label_map[l] for l in pred_labels]

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary"
    )
    accuracy = sum(g == p for g, p in zip(gold_labels, pred_labels)) / len(results)

    print("\n" + "=" * 50)
    print("ASHA-AI EVALUATION RESULTS")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["HOME", "REFER"]))
    print(f"Overall accuracy : {accuracy:.1%}")
    print(f"REFER precision  : {p:.1%}  (of predicted REFER, how many were correct)")
    print(f"REFER recall     : {r:.1%}  ← KEY METRIC (missed referrals = patient harm)")
    print(f"REFER F1 score   : {f1:.1%}")

    summary = {
        "accuracy":         round(accuracy, 4),
        "refer_precision":  round(p, 4),
        "refer_recall":     round(r, 4),
        "refer_f1":         round(f1, 4),
        "n_examples":       len(results),
        "results":          results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Results saved → {output_path}")

    return summary


if __name__ == "__main__":
    run_evaluation()
