# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASHA-AI  |  data_builder.py
# Builds and saves the fine-tuning dataset in Gemma chat-template format
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json
from pathlib import Path
from typing import List, Dict

from .config import RAW_DATA, SYSTEM_PROMPT, DATA_REPEAT


def format_for_gemma4(item: Dict) -> Dict:
    """
    Format a Q&A pair into Gemma 4 chat template + instruction/output fields.
    """
    return {
        "instruction": item["q"],
        "output": item["a"],
        "text": (
            f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
            f"<start_of_turn>user\n{item['q']}<end_of_turn>\n"
            f"<start_of_turn>model\n{item['a']}<end_of_turn>"
        ),
    }


def build_dataset(
    raw_data: List[Dict] = RAW_DATA,
    repeat: int = DATA_REPEAT,
    output_dir: str = "data",
    train_ratio: float = 0.8,
) -> Dict:
    """
    Build and save train/eval JSONL files.

    Args:
        raw_data:    List of {"q": ..., "a": ...} dicts.
        repeat:      Number of times to duplicate each example (augmentation).
        output_dir:  Directory to write train.jsonl / eval.jsonl.
        train_ratio: Fraction of unique examples to use as train base.

    Returns:
        Dict with keys "train_path", "eval_path", "n_train", "n_eval".
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    formatted_all = [format_for_gemma4(d) for d in raw_data]
    augmented     = formatted_all * repeat

    split     = int(len(raw_data) * train_ratio)
    eval_data = [format_for_gemma4(d) for d in raw_data[split:]]

    train_path = Path(output_dir) / "train.jsonl"
    eval_path  = Path(output_dir) / "eval.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in augmented:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Dataset ready:")
    print(f"   Train : {len(augmented)} examples → {train_path}")
    print(f"   Eval  : {len(eval_data)} examples → {eval_path}")
    print(f"\nSample entry:\n{augmented[0]['text'][:300]}...\n")

    return {
        "train_path": str(train_path),
        "eval_path":  str(eval_path),
        "n_train":    len(augmented),
        "n_eval":     len(eval_data),
    }


if __name__ == "__main__":
    build_dataset()
