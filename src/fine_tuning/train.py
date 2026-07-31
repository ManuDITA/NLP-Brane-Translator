"""
train.py

QLoRA fine-tuning of a Qwen model on BraneScript (intent → code) pairs.

Two training modes
──────────────────
  sft   Supervised Fine-Tuning on (intent → BraneScript) pairs.
        Classic next-token prediction. Fast, stable, good starting point.

  grpo  Group Relative Policy Optimization with execution-based reward.
        The model generates multiple BraneScripts per intent; each is run on
        a live Brane instance and scored. The model learns from the relative
        quality of scripts within each group — no reference answer needed.
        Requires a Brane instance accessible via `brane workflow run`.

Output directories are named per model so multiple runs co-exist:
  src/fine_tuning/output_{slug}/        — SFT LoRA adapter
  src/fine_tuning/output_{slug}_grpo/   — GRPO LoRA adapter
  src/fine_tuning/output_merged_{slug}/ — merged full model

Usage
─────
  # SFT (auto-resumes from latest checkpoint)
  python src/fine_tuning/train.py
  python src/fine_tuning/train.py --mode sft

  # GRPO (requires Brane instance at $BRANE_INSTANCE, default: local-instance)
  python src/fine_tuning/train.py --mode grpo

  # GRPO warm-started from SFT adapter (recommended: run --merge first)
  python src/fine_tuning/train.py --mode grpo --warm-start

  # Ignore existing checkpoints
  python src/fine_tuning/train.py --mode sft --restart

  # Merge LoRA adapter into base model
  python src/fine_tuning/train.py --merge [--mode grpo]

  # Test GRPO training loop without a real Brane instance
  python src/fine_tuning/train.py --mode grpo --mock-rewards

Environment variables
─────────────────────
  FINETUNE_MODEL    HuggingFace model ID  (default: Qwen/Qwen3.5-9B)
  BRANE_INSTANCE    Brane instance name   (default: local-instance)

Dependencies
────────────
  pip install trl peft accelerate bitsandbytes datasets transformers
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL  = os.environ.get("FINETUNE_MODEL", "Qwen/Qwen3.5-9B")
MODEL_SLUG  = BASE_MODEL.split("/")[-1].lower()          # e.g. "qwen3.5-9b"

_HERE       = Path(__file__).resolve().parent
_ROOT       = _HERE.parent.parent                         # project root
_MODELS_DIR = _ROOT / "outputs" / "models"

OUTPUT_DIR  = _MODELS_DIR / f"output_{MODEL_SLUG}"        # SFT adapter
GRPO_DIR    = _MODELS_DIR / f"output_{MODEL_SLUG}_grpo"   # GRPO adapter
MERGED_DIR  = _MODELS_DIR / f"output_merged_{MODEL_SLUG}" # merged full model

TRAIN_FILE  = _ROOT / "data" / "training" / "train.jsonl"
VAL_FILE    = _ROOT / "data" / "training" / "val.jsonl"

# SFT hyperparameters
MAX_SEQ_LEN   = 2048
LORA_RANK     = 16
SFT_EPOCHS    = 3
BATCH_SIZE    = 2
GRAD_ACCUM    = 4
SFT_LR        = 2e-4
WARMUP_STEPS  = 10
SAVE_STEPS    = 25
LOGGING_STEPS = 5

# GRPO hyperparameters
GRPO_EPOCHS          = 2
GRPO_LR              = 5e-6
GRPO_BATCH_SIZE      = 1
GRPO_GRAD_ACCUM      = 8
GRPO_NUM_GENERATIONS = 8     # completions per prompt; reduce to 4 for 27B+
GRPO_MAX_NEW_TOKENS  = 512
GRPO_TEMPERATURE     = 0.8
GRPO_TOP_P           = 0.9
GRPO_WARMUP_STEPS    = 5

# Brane executor
BRANE_INSTANCE  = os.environ.get("BRANE_INSTANCE", "local-instance")
BRANE_TIMEOUT   = 30          # seconds per script execution
BRANE_WORKERS   = 8           # parallel execution threads


# ---------------------------------------------------------------------------
# Brane reward function
# ---------------------------------------------------------------------------

# Reference stdout lookup: intent → expected stdout from execution_results.jsonl
# Built once at module load time; used by the reward function during GRPO.
_REFERENCE_OUTPUTS: dict[str, str] = {}

def _load_reference_outputs() -> None:
    """Load pre-executed reference stdout values from execution_results.jsonl."""
    ref_file = _HERE.parent.parent / "data" / "training" / "execution_results.jsonl"
    if not ref_file.exists():
        print(f"⚠️  Reference outputs not found at {ref_file} — stdout matching disabled.")
        return
    loaded = 0
    with open(ref_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("success") and r.get("intent"):
                    _REFERENCE_OUTPUTS[r["intent"]] = (r.get("stdout") or "").strip()
                    loaded += 1
            except Exception:
                pass
    print(f"📚 Loaded {loaded} reference outputs for stdout-matching reward.")

_load_reference_outputs()


def _extract_branescript(text: str) -> str:
    """Strip thinking blocks and markdown fences from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _score_one(code: str, reference_stdout: str = "") -> float:
    """
    Run a BraneScript on the Brane instance and return a reward score.

    Scoring uses stdout comparison when a reference is available:
      +1.0  ran successfully AND stdout matches reference
      +0.5  ran successfully, no reference stdout to compare (e.g. commit_result only)
      +0.3  ran successfully but stdout differs from reference
      -0.5  runtime error (compiled but crashed)
      -1.0  compilation / syntax error, timeout, or empty code
    """
    code = _extract_branescript(code)
    if not code or len(code) < 10:
        return -1.0

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".bs", encoding="utf-8", delete=False
    ) as f:
        f.write(code)
        path = f.name

    try:
        result = subprocess.run(
            ["brane", "workflow", "run", BRANE_INSTANCE, path],
            capture_output=True, text=True, timeout=BRANE_TIMEOUT,
        )

        if result.returncode != 0:
            stderr = result.stderr.lower()
            if ("compilation of workflow failed" in stderr
                    or "parse error" in stderr
                    or "does not exist" in stderr
                    or "undefined function" in stderr):
                return -1.0   # syntax / compile error
            return -0.5       # runtime error (compiled, crashed at execution)

        # Script ran successfully — compare stdout to reference
        stdout = result.stdout.strip()

        if not reference_stdout:
            # No reference available (e.g. script only calls commit_result)
            return 0.5

        if stdout == reference_stdout:
            return 1.0    # exact match

        return 0.3        # ran but different output

    except subprocess.TimeoutExpired:
        return -1.0
    except FileNotFoundError:
        raise RuntimeError(
            "`brane` not found in PATH. "
            "GRPO requires a running Brane instance. "
            "Use --mock-rewards to test the training loop without one."
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def brane_execution_reward(completions: list[str], prompts=None, **kwargs) -> list[float]:
    """
    TRL-compatible reward function with stdout-matching.

    For each completion:
      1. Extract the intent from the prompt (user message)
      2. Look up the reference stdout from execution_results.jsonl
      3. Run the generated BraneScript on Brane
      4. Score based on execution success + stdout match
    """
    # Extract intent strings from prompt message lists
    refs: list[str] = []
    if prompts:
        for prompt in prompts:
            if isinstance(prompt, list):
                intent = next(
                    (m["content"] for m in prompt if m.get("role") == "user"), ""
                )
            else:
                intent = str(prompt)
            refs.append(_REFERENCE_OUTPUTS.get(intent, ""))
    else:
        refs = [""] * len(completions)

    rewards: list[float | None] = [None] * len(completions)
    with ThreadPoolExecutor(max_workers=min(BRANE_WORKERS, len(completions))) as ex:
        futures = {
            ex.submit(_score_one, code, ref): i
            for i, (code, ref) in enumerate(zip(completions, refs))
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                rewards[idx] = fut.result()
            except Exception as e:
                print(f"  ⚠️  reward error for completion {idx}: {e}", flush=True)
                rewards[idx] = -1.0
    return rewards


def mock_reward(completions: list[str], **kwargs) -> list[float]:
    """Dummy reward for testing the GRPO loop without a Brane instance."""
    import random
    return [random.choice([-1.0, -0.5, 0.3, 0.5, 1.0]) for _ in completions]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_sft_samples(samples: list[dict], tokenizer) -> list[dict]:
    """Apply chat template to full (system+user+assistant) messages for SFT."""
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


def format_grpo_prompts(samples: list[dict]) -> list[dict]:
    """Extract system+user messages only (no assistant) for GRPO prompts."""
    return [
        {"prompt": [m for m in s["messages"] if m["role"] != "assistant"]}
        for s in samples
    ]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def latest_checkpoint(output_dir: Path):
    if not output_dir.exists():
        return None
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


# ---------------------------------------------------------------------------
# SFT training
# ---------------------------------------------------------------------------

def train_sft(restart: bool = False):
    print(f"🔧 Mode       : SFT")
    print(f"🔧 Base model : {BASE_MODEL}")
    print(f"📂 Output dir : {OUTPUT_DIR}")
    print(f"📊 Train file : {TRAIN_FILE}  ({sum(1 for _ in open(TRAIN_FILE))} samples)")

    if not TRAIN_FILE.exists():
        print(f"❌ {TRAIN_FILE} not found — run prepare_dataset.py first.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    resume_from = None
    if not restart:
        resume_from = latest_checkpoint(OUTPUT_DIR)
        if resume_from:
            print(f"🔄 Resuming from: {resume_from}")
        else:
            print("🆕 Starting fresh")
    else:
        print("🔁 --restart: ignoring existing checkpoints")

    try:
        from unsloth import FastLanguageModel
        _sft_unsloth(resume_from)
        return
    except ImportError:
        print("ℹ️  unsloth not available — using plain trl + peft + bitsandbytes")
    except Exception as exc:
        print(f"⚠️  unsloth failed ({exc}) — falling back to plain HF")

    _sft_plain(resume_from)


def _build_sft_config(resume_from):
    from trl import SFTConfig
    return SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=SFT_LR,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataset_text_field="text",
        # Do NOT set packing=True — responses are variable length and packing
        # would concatenate multiple examples, breaking response masking.
        packing=False,
    )


def _sft_unsloth(resume_from):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from datasets import Dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2, lora_dropout=0.05, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    train_ds = Dataset.from_list(format_sft_samples(load_jsonl(TRAIN_FILE), tokenizer))
    val_ds   = Dataset.from_list(format_sft_samples(load_jsonl(VAL_FILE),   tokenizer))
    print(f"✅ Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
        max_seq_length=MAX_SEQ_LEN, args=_build_sft_config(resume_from),
    )

    # Response-only training: only compute loss on the assistant reply tokens,
    # not on the system prompt or user message. This gives the model a much
    # stronger signal to learn BraneScript AND to stop with <|im_end|>.
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        print("✅ Response-only training enabled (unsloth)")
    except Exception as exc:
        print(f"⚠️  train_on_responses_only not available: {exc} — training on full sequence")

    trainer.train(resume_from_checkpoint=resume_from)

    print(f"\n💾 Saving SFT adapter → {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("✅ Done.")


def _sft_plain(resume_from):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    print("📥 Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token  = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("📥 Loading model in 4-bit…")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    train_ds = Dataset.from_list(format_sft_samples(load_jsonl(TRAIN_FILE), tokenizer))
    val_ds   = Dataset.from_list(format_sft_samples(load_jsonl(VAL_FILE),   tokenizer))
    print(f"✅ Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # Response-only training: mask system+user tokens from the loss so the
    # model only learns to predict the assistant (BraneScript) tokens.
    # This gives a clean gradient signal on the code AND on <|im_end|> (stop).
    try:
        from trl import DataCollatorForCompletionOnlyLM
        # <|im_start|>assistant\n marks the start of every assistant reply in ChatML
        response_template_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=tokenizer
        )
        print("✅ Response-only training enabled (DataCollatorForCompletionOnlyLM)")
    except Exception as exc:
        print(f"⚠️  DataCollatorForCompletionOnlyLM not available: {exc} — training on full sequence")
        collator = None

    SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
        max_seq_length=MAX_SEQ_LEN,
        data_collator=collator,
        args=_build_sft_config(resume_from),
    ).train(resume_from_checkpoint=resume_from)

    print(f"\n💾 Saving SFT adapter → {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("✅ Done.")


# ---------------------------------------------------------------------------
# GRPO training
# ---------------------------------------------------------------------------

def train_grpo(restart: bool = False, warm_start: bool = False,
               mock_rewards: bool = False):
    """
    GRPO training with Brane execution as the reward signal.

    For each training step:
      1. Sample a batch of intents from train.jsonl
      2. Generate GRPO_NUM_GENERATIONS BraneScripts per intent
      3. Run all generated scripts on Brane in parallel
      4. Score each: +1.0 (runs), +0.3 (runtime error), -1.0 (compile/timeout)
      5. Normalise rewards within each group (this is the R in GRPO)
      6. Update LoRA adapters to increase probability of higher-reward scripts

    Warm-start (--warm-start):
      Loads output_merged_{slug}/ as the base model instead of the raw
      pretrained model. Run `--merge` after SFT before using this flag.
    """
    base = str(MERGED_DIR) if warm_start else BASE_MODEL
    if warm_start and not MERGED_DIR.exists():
        print(f"❌ --warm-start requested but {MERGED_DIR} not found.")
        print("   Run: python train.py --merge   first.")
        sys.exit(1)

    reward_fn = mock_reward if mock_rewards else brane_execution_reward

    print(f"🔧 Mode          : GRPO {'(mock rewards)' if mock_rewards else ''}")
    print(f"🔧 Base model    : {base}")
    print(f"📂 Output dir    : {GRPO_DIR}")
    print(f"🎯 Reward        : {'mock (random)' if mock_rewards else f'brane execution ({BRANE_INSTANCE})'}")
    print(f"👥 Generations   : {GRPO_NUM_GENERATIONS} per prompt")
    print(f"📊 Train prompts : {TRAIN_FILE}")

    if not TRAIN_FILE.exists():
        print(f"❌ {TRAIN_FILE} not found — run prepare_dataset.py first.")
        sys.exit(1)

    GRPO_DIR.mkdir(parents=True, exist_ok=True)

    resume_from = None
    if not restart:
        resume_from = latest_checkpoint(GRPO_DIR)
        if resume_from:
            print(f"🔄 Resuming from: {resume_from}")
        else:
            print("🆕 Starting fresh")
    else:
        print("🔁 --restart: ignoring existing checkpoints")

    _grpo_plain(base, reward_fn, resume_from)


def _build_grpo_config(resume_from):
    from trl import GRPOConfig
    return GRPOConfig(
        output_dir=str(GRPO_DIR),
        num_train_epochs=GRPO_EPOCHS,
        per_device_train_batch_size=GRPO_BATCH_SIZE,
        gradient_accumulation_steps=GRPO_GRAD_ACCUM,
        learning_rate=GRPO_LR,
        warmup_steps=GRPO_WARMUP_STEPS,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to="none",
        # GRPO-specific
        num_generations=GRPO_NUM_GENERATIONS,
        max_new_tokens=GRPO_MAX_NEW_TOKENS,
        temperature=GRPO_TEMPERATURE,
        top_p=GRPO_TOP_P,
        # Keep generations short enough to avoid OOM
        max_prompt_length=512,
    )


def _grpo_plain(base_model: str, reward_fn, resume_from):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import GRPOTrainer
    from datasets import Dataset

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    print("📥 Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # GRPOTrainer needs left-padding

    print("📥 Loading model in 4-bit…")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    # GRPO only needs prompts — no reference answers
    train_ds = Dataset.from_list(format_grpo_prompts(load_jsonl(TRAIN_FILE)))
    print(f"✅ GRPO train prompts: {len(train_ds)}")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=_build_grpo_config(resume_from),
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    print("\n🏋️  GRPO training…")
    print("   Each step: generate → execute on Brane → score → update")
    trainer.train(resume_from_checkpoint=resume_from)

    print(f"\n💾 Saving GRPO adapter → {GRPO_DIR}")
    model.save_pretrained(str(GRPO_DIR))
    tokenizer.save_pretrained(str(GRPO_DIR))
    print("✅ Done.")


# ---------------------------------------------------------------------------
# Merge LoRA into base model
# ---------------------------------------------------------------------------

def merge(mode: str = "sft"):
    """Merge LoRA adapter into base weights → standalone HF model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    adapter_dir = GRPO_DIR if mode == "grpo" else OUTPUT_DIR
    base        = str(MERGED_DIR) if (mode == "grpo" and MERGED_DIR.exists()) else BASE_MODEL

    if not adapter_dir.exists():
        print(f"❌ Adapter not found at {adapter_dir} — train first.")
        sys.exit(1)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🔧 Loading base  : {base}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    print(f"🔧 Loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    print("🔀 Merging weights…")
    model = model.merge_and_unload()

    print(f"💾 Saving merged model → {MERGED_DIR}")
    model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"✅ Merged model saved to {MERGED_DIR}")
    print(f"   Use with evaluate.py: --model {MERGED_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraneScript fine-tuning: SFT or GRPO")
    parser.add_argument(
        "--mode", choices=["sft", "grpo"], default="sft",
        help="Training mode: sft (supervised) or grpo (execution-reward RL)",
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Ignore existing checkpoints and start training from scratch",
    )
    parser.add_argument(
        "--warm-start", action="store_true",
        help="(GRPO only) Start from the merged SFT model instead of raw base",
    )
    parser.add_argument(
        "--mock-rewards", action="store_true",
        help="(GRPO only) Use random rewards instead of real Brane execution",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge LoRA adapter into base model and save a full HF checkpoint",
    )
    args = parser.parse_args()

    print(f"Model : {BASE_MODEL}  (slug: {MODEL_SLUG})")

    if args.merge:
        merge(mode=args.mode)
    elif args.mode == "grpo":
        train_grpo(
            restart=args.restart,
            warm_start=args.warm_start,
            mock_rewards=args.mock_rewards,
        )
    else:
        train_sft(restart=args.restart)


