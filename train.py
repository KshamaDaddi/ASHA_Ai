# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASHA-AI  |  train.py
# Fine-tune Gemma 3 4B on ASHA triage data using Unsloth + LoRA + SFTTrainer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import torch
from pathlib import Path

from .config import (
    MODEL_NAME, ADAPTER_DIR, MAX_SEQ_LEN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, GRAD_ACCUM_STEPS,
    LEARNING_RATE, WARMUP_RATIO,
)
from .data_builder import build_dataset


def load_base_model():
    """Load the 4-bit quantised Gemma 3 4B base model with Unsloth."""
    from unsloth import FastLanguageModel

    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    print("✅ Base model loaded")
    return model, tokenizer


def add_lora(model):
    """Attach LoRA adapters to the base model."""
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"✅ LoRA adapters: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.1f}%)"
    )
    return model


def fine_tune(
    data_dir: str = "data",
    adapter_dir: str = ADAPTER_DIR,
) -> dict:
    """
    Full fine-tuning pipeline:
      1. Build dataset
      2. Load + patch model
      3. Train
      4. Save adapter

    Returns training stats dict.
    """
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    dataset_info = build_dataset(output_dir=data_dir)

    # ── 2. Model ───────────────────────────────────────────────────────────────
    model, tokenizer = load_base_model()
    model = add_lora(model)

    dataset = load_dataset(
        "json",
        data_files={"train": dataset_info["train_path"]},
        split="train",
    )
    print(f"✅ Dataset loaded: {len(dataset)} examples")

    # ── 3. Train ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=TrainingArguments(
            output_dir=adapter_dir,
            num_train_epochs=TRAIN_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            warmup_ratio=WARMUP_RATIO,
            learning_rate=LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=20,
            save_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            report_to="none",
        ),
    )

    print("\n🚀 Starting fine-tuning (~60–90 min on T4)…")
    stats = trainer.train()

    # ── 4. Save ────────────────────────────────────────────────────────────────
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"\n✅ Fine-tuning complete!")
    print(f"   Loss   : {stats.training_loss:.4f}")
    print(f"   Adapter: ./{adapter_dir}/")

    return {
        "training_loss": stats.training_loss,
        "adapter_dir":   adapter_dir,
    }


if __name__ == "__main__":
    fine_tune()
