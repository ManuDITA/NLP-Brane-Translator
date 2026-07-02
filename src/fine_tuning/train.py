"""
train.py

QLoRA fine-tuning of a Qwen model on BraneScript (intent → code) pairs.

Tries to use unsloth for fast training; falls back to plain trl + peft +
bitsandbytes if unsloth is not available (e.g. on Snellius with CUDA 13.x).

Resume support: re-running train.py automatically continues from the latest
checkpoint in OUTPUT_DIR.  Use --restart to ignore checkpoints and start fresh.

Usage:
    # Train (auto-resumes if a checkpoint exists)
    python src/fine_tuning/train.py

    # Train from scratch, ignoring any existing checkpoints
    python src/fine_tuning/train.py --restart

    # Merge LoRA weights into the base model and save a full HF checkpoint
    python src/fine_tuning/train.py --merge

Dependencies (plain, no unsloth):
    pip install trl peft accelerate bitsandbytes datasets transformers
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL  = os.environ.get("FINETUNE_MODEL", "Qwen/Qwen3-8B")
OUTPUT_DIR  = Path(__file__).resolve().parent / "output"
MERGED_DIR  = Path(__file__).resolve().parent / "output_merged"
TRAIN_FILE  = Path(__file__).resolve().parent / "train.jsonl"
VAL_FILE    = Path(__file__).resolve().parent / "val.jsonl"

MAX_SEQ_LEN  = 2048
LORA_RANK    = 16
EPOCHS       = 3
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
LR           = 2e-4
WARMUP_STEPS = 10
SAVE_STEPS   = 25   # checkpoint every 25 steps (fine-grained resume)
LOGGING_STEPS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_samples(samples: list[dict], tokenizer) -> list[dict]:
    """Apply chat template to each sample (list of messages)."""
    out = []
    for s in samples:
        try:
            text = tokenizer.apply_chat_template(
                s["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                s["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        out.append({"text": text})
    return out


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def latest_checkpoint(output_dir: Path):
    """Return path to the latest checkpoint, or None."""
    if not output_dir.exists():
        return None
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


# ---------------------------------------------------------------------------
# Training — tries unsloth, falls back to plain HF
# ---------------------------------------------------------------------------
def train(restart: bool = False):
    print(f"🔧 Base model : {BASE_MODEL}")
    print(f"📂 Output dir : {OUTPUT_DIR}")
    print(f"📊 Train file : {TRAIN_FILE}  ({sum(1 for _ in open(TRAIN_FILE))} samples)")

    if not TRAIN_FILE.exists():
        print(f"❌ {TRAIN_FILE} not found.")
        print("   Run: python src/harvest_training_data.py --exclude-benchmark")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Determine resume checkpoint ─────────────────────────────────────
    resume_from = None
    if not restart:
        resume_from = latest_checkpoint(OUTPUT_DIR)
        if resume_from:
            print(f"🔄 Resuming from checkpoint: {resume_from}")
        else:
            print("🆕 No checkpoint found — starting fresh")
    else:
        print("🔁 --restart flag set — ignoring any existing checkpoints")

    # ── Try unsloth first ────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
        _train_unsloth(resume_from)
        return
    except ImportError:
        print("ℹ️  unsloth not available — using plain trl + peft + bitsandbytes")
    except Exception as exc:
        print(f"⚠️  unsloth failed ({exc}) — falling back to plain HF")

    _train_plain(resume_from)


def _build_trainer_args(resume_from):
    from trl import SFTConfig
    return SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        bf16=True,           # H100 supports bf16; falls back gracefully
        fp16=False,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,          # keep only 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataset_text_field="text",
    )


def _train_unsloth(resume_from):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from datasets import Dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_dataset = Dataset.from_list(format_samples(load_jsonl(TRAIN_FILE), tokenizer))
    val_dataset   = Dataset.from_list(format_samples(load_jsonl(VAL_FILE),   tokenizer))
    print(f"✅ Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        max_seq_length=MAX_SEQ_LEN,
        args=_build_trainer_args(resume_from),
    )

    print("\n🏋️  Fine-tuning (unsloth)...")
    trainer.train(resume_from_checkpoint=resume_from)

    print(f"\n💾 Saving LoRA adapter → {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("✅ Done.")


def _train_plain(resume_from):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("📥 Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("📥 Loading model in 4-bit…")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    train_dataset = Dataset.from_list(format_samples(load_jsonl(TRAIN_FILE), tokenizer))
    val_dataset   = Dataset.from_list(format_samples(load_jsonl(VAL_FILE),   tokenizer))
    print(f"✅ Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        max_seq_length=MAX_SEQ_LEN,
        args=_build_trainer_args(resume_from),
    )

    print("\n🏋️  Fine-tuning (plain HF)…")
    trainer.train(resume_from_checkpoint=resume_from)

    print(f"\n💾 Saving LoRA adapter → {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("✅ Done.")


# ---------------------------------------------------------------------------
# Merge LoRA into base model (produces a standalone HF model)
# ---------------------------------------------------------------------------
def merge():
    """
    Merge the LoRA adapter into the base weights and save a full HF model.
    Use this before running evaluate.py with the fine-tuned model.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    if not OUTPUT_DIR.exists():
        print(f"❌ {OUTPUT_DIR} not found — run training first.")
        sys.exit(1)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🔧 Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"🔧 Loading LoRA adapter: {OUTPUT_DIR}")
    model = PeftModel.from_pretrained(base, str(OUTPUT_DIR))

    print("🔀 Merging weights…")
    model = model.merge_and_unload()

    print(f"💾 Saving merged model → {MERGED_DIR}")
    model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"✅ Merged model saved to {MERGED_DIR}")
    print(f"   Use this path with evaluate.py: --model {MERGED_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for BraneScript")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore existing checkpoints and start training from scratch")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA adapter into base model and save a full HF checkpoint")
    args = parser.parse_args()

    if args.merge:
        merge()
    else:
        train(restart=args.restart)

