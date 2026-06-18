"""
train.py

QLoRA fine-tuning of a Qwen model on BraneScript (intent → code) pairs.
Uses unsloth for efficient 4-bit training on a single GPU.

Requirements:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes

Run (after running prepare_dataset.py):
    python fine_tuning/train.py

After training, export to GGUF:
    python fine_tuning/train.py --export
"""

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — adjust to your hardware and model
# ---------------------------------------------------------------------------
BASE_MODEL   = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # or Qwen3-4B, Qwen3-8B, etc.
OUTPUT_DIR   = Path(__file__).resolve().parent / "output"
TRAIN_FILE   = Path(__file__).resolve().parent / "train.jsonl"
VAL_FILE     = Path(__file__).resolve().parent / "val.jsonl"
GGUF_DIR     = Path(__file__).resolve().parent / "gguf"

MAX_SEQ_LEN  = 2048
LORA_RANK    = 16
EPOCHS       = 3
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
LR           = 2e-4
WARMUP_STEPS = 10


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        print("                 pip install --no-deps trl peft accelerate bitsandbytes datasets")
        raise

    print(f"🔧 Loading base model: {BASE_MODEL}")
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

    def load_jsonl(path: Path) -> list[dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def format_sample(sample: dict) -> dict:
        """Apply the tokenizer chat template to each (messages) sample."""
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    print(f"📂 Loading dataset from {TRAIN_FILE} / {VAL_FILE}")
    train_data = [format_sample(s) for s in load_jsonl(TRAIN_FILE)]
    val_data   = [format_sample(s) for s in load_jsonl(VAL_FILE)]

    train_dataset = Dataset.from_list(train_data)
    val_dataset   = Dataset.from_list(val_data)

    print(f"✅ Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_steps=WARMUP_STEPS,
            lr_scheduler_type="cosine",
            fp16=True,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
        ),
    )

    print("\n🏋️  Starting fine-tuning...")
    trainer.train()

    print(f"\n💾 Saving LoRA adapter to {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("✅ Training complete.")


# ---------------------------------------------------------------------------
# GGUF export (for Ollama)
# ---------------------------------------------------------------------------

def export_gguf():
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        raise

    print(f"🔧 Loading fine-tuned model from {OUTPUT_DIR}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(OUTPUT_DIR),
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    gguf_path = str(GGUF_DIR / "branescript-qwen-q4_k_m")

    print(f"📦 Exporting to GGUF (Q4_K_M) at {gguf_path}")
    model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
    print(f"✅ GGUF saved to {gguf_path}")
    print(f"\nNext step: register with Ollama using fine_tuning/Modelfile.template")
    print(f"  ollama create branescript-qwen -f fine_tuning/Modelfile.template")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true",
                        help="Export the fine-tuned model to GGUF for Ollama (run after training)")
    args = parser.parse_args()

    if args.export:
        export_gguf()
    else:
        train()
