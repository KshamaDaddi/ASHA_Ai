"""
finetune_gemma4.py
==================
Fine-tunes Gemma 4 E4B using Unsloth on a Kaggle T4/P100 GPU.
Runtime: ~2.5 hours on T4 for 2000 examples.

Run this as a Kaggle Notebook (GPU accelerator ON).
After training, publish your weights to Kaggle Datasets or HuggingFace Hub.

pip install unsloth[kaggle-new] xformers trl peft datasets
"""

# ─── 1. Install (run this cell first in Kaggle) ───────────────────────────────
# !pip install unsloth[kaggle-new] xformers==0.0.28.post3 trl peft datasets -q

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# ─── 2. Config ────────────────────────────────────────────────────────────────
MODEL_ID     = "unsloth/gemma-4-it-4b-unsloth-bnb-4bit"   # Gemma 4 E4B quantised
MAX_SEQ_LEN  = 2048
LORA_RANK    = 16
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
EPOCHS       = 3
LR           = 2e-4
OUTPUT_DIR   = "asha_ai_gemma4_adapter"

# ─── 3. Load base model with 4-bit quantisation ───────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,           # auto: float16 on T4, bfloat16 on A100
    load_in_4bit=True,
)

# ─── 4. Add LoRA adapters ─────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK * 2,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # saves VRAM on Kaggle
    random_state=42,
)

print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ─── 5. Load dataset ──────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files={"train": "data/asha_instructions.jsonl"})
print(f"Training examples: {len(dataset['train'])}")

# ─── 6. Trainer ───────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_ratio=0.1,
    learning_rate=LR,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    report_to="none",     # set to "wandb" if you want live loss curves
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=training_args,
)

trainer.train()

# ─── 7. Save adapter + push to HuggingFace (optional) ────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to ./{OUTPUT_DIR}/")

# Uncomment to push weights publicly (required for hackathon submission):
# model.push_to_hub("your-hf-username/asha-ai-gemma4-e4b", token="hf_xxx")
# tokenizer.push_to_hub("your-hf-username/asha-ai-gemma4-e4b", token="hf_xxx")

# ─── 8. Quick inference test ──────────────────────────────────────────────────
FastLanguageModel.for_inference(model)

SYSTEM = (
    "You are ASHA-AI, a trusted health assistant for ASHA workers in rural India. "
    "Provide clear, actionable triage guidance. When in doubt, advise referral."
)
query = "A 6-month-old baby has high fever and fits (convulsions). What should I do?"

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": query},
]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=256, temperature=0.2, do_sample=True)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print("\n--- ASHA-AI response ---")
print(response)
