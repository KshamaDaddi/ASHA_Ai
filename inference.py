# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASHA-AI  |  inference.py
# Load the fine-tuned Gemma adapter and run RAG-augmented triage inference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import torch

from .config import ADAPTER_DIR, MAX_SEQ_LEN, SYSTEM_PROMPT
from .knowledge_base import retrieve_context

# Module-level singletons — loaded once per process
_model     = None
_tokenizer = None


def load_model(adapter_dir: str = ADAPTER_DIR) -> tuple:
    """
    Load the fine-tuned LoRA adapter (or base model if adapter not found)
    and switch to fast inference mode.

    Returns (model, tokenizer).
    """
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    from unsloth import FastLanguageModel
    from pathlib import Path

    model_path = adapter_dir if Path(adapter_dir).exists() else adapter_dir
    print(f"Loading model from: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("✅ Model ready for inference")

    _model     = model
    _tokenizer = tokenizer
    return _model, _tokenizer


def asha_triage(
    query: str,
    language: str = "en",
    show_context: bool = False,
) -> str:
    """
    Run a single triage query through the RAG + fine-tuned Gemma pipeline.

    Args:
        query:        The symptom description in English.
        language:     "en" for English responses, "kn" for Kannada, etc.
        show_context: If True, print the retrieved RAG context to stdout.

    Returns:
        The model's raw response string.
    """
    model, tokenizer = load_model()
    context = retrieve_context(query)

    lang_note = (
        "Respond in Kannada (ಕನ್ನಡ)." if language == "kn"
        else "Respond in simple English."
    )
    prompt = (
        f"Relevant ASHA guidelines:\n{context}\n\n"
        f"ASHA worker question: {query}\n\n"
        f"{lang_note} Give a numbered, step-by-step response. "
        f"Start with the most urgent action."
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "text", "text": prompt}]},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=350,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    if show_context:
        print("📚 Retrieved context:")
        print(context[:400], "…\n")

    return response
