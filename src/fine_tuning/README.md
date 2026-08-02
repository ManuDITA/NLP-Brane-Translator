# Fine-tuning BraneScript — Step by Step

## Overview

Fine-tuning teaches your local Qwen model to generate BraneScript natively,
so it produces correct code reliably without extensive prompt engineering.

**Workflow:**
1. Generate more examples → `src/example_generator.py`
2. Prepare training data → `fine_tuning/prepare_dataset.py`
3. Fine-tune the model → `fine_tuning/train.py`
4. Export to GGUF → `fine_tuning/train.py --export`
5. Register in Ollama → `ollama create`
6. Use in pipeline → update `pipeline.py` model name

---

## Requirements

A GPU with at least **12 GB VRAM** is recommended (16+ GB for the 7B model).
CPU-only training is possible but very slow.

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes datasets
```

---

## Step 1 — Grow your example library

The more examples you provide, the better the fine-tuned model will be.

Run the example generator to automatically produce more (intent, BraneScript) pairs:

```bash
cd NLP-Brane-Translator
python src/example_generator.py
```

Or add examples manually to the `.jsonl` files in `data/examples/`.
Each line must be:
```json
{"intent": "...", "branescript": "..."}
```

---

## Step 2 — Prepare the dataset

```bash
python fine_tuning/prepare_dataset.py
```

This creates:
- `data/training/train.jsonl`
- `data/training/val.jsonl`

---

## Step 3 — Fine-tune

```bash
python fine_tuning/train.py
```

Training will save the LoRA adapter to `fine_tuning/output/`.
Adjust the constants at the top of `train.py` to match your hardware:
- `BASE_MODEL` — Qwen model to use (default: `Qwen2.5-7B-Instruct`)
- `EPOCHS`, `BATCH_SIZE`, `LR` — training hyper-parameters

---

## Step 4 — Export to GGUF

```bash
python fine_tuning/train.py --export
```

The quantised model lands at `fine_tuning/gguf/branescript-qwen-q4_k_m.gguf`.

---

## Step 5 — Register in Ollama

```bash
ollama create branescript-qwen -f fine_tuning/Modelfile.template
```

Test it:
```bash
ollama run branescript-qwen "Analyze heart disease risk for a 55-year-old male with blood pressure 150"
```

---

## Step 6 — Use in the pipeline

In `src/pipeline.py`, change the model name:

```python
llm = Ollama(
    model="branescript-qwen",   # ← your fine-tuned model
    temperature=0.4,
    ...
)
```

---

## Tips

- **More data = better results.** Aim for 200+ diverse examples.
- Cover all language features: basic vars, control flow, functions, classes, parallel, data, packages.
- Include edge cases: multi-step workflows, nested if/else, parallel + site attributes.
- After fine-tuning, re-run `python src/knowledgeBase.py` to rebuild the vector DB with the latest examples.
