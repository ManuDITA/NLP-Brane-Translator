# NLP → BraneScript Translator — Source Code

This folder contains the Python source code for the NLP-to-BraneScript translation pipeline. The system takes a plain-English user intent and produces a valid BraneScript workflow ready to run on the Brane Framework.

---

## Table of Contents

1. [Architecture overview](#architecture-overview)
2. [How the pipeline works step by step](#how-the-pipeline-works-step-by-step)
3. [What changed and why](#what-changed-and-why)
4. [Files in this folder](#files-in-this-folder)
5. [Quick start](#quick-start)
6. [Running the pipeline](#running-the-pipeline)
7. [Rebuilding the knowledge bases](#rebuilding-the-knowledge-bases)
8. [Generating more training examples](#generating-more-training-examples)
9. [Configuration reference](#configuration-reference)

---

## Architecture overview

```
User intent (plain English)
        │
        ▼
┌─────────────────────────┐
│   IntentDecomposer      │  breaks intent into BraneScript sub-tasks
│   (intent_decomposer.py)│  retrieves relevant language spec chunks from DB
└───────────┬─────────────┘
            │ subtasks + lang_context
            ▼
┌─────────────────────────┐
│   PkgRetriever          │  finds the most relevant package / dataset docs
│   (pkg_retriever.py)    │  from the package vector DB
└───────────┬─────────────┘
            │ pkg_context
            ▼
┌─────────────────────────┐
│   Prompt assembly       │  combines: few-shot examples + sub-tasks +
│   (pipeline.py)         │  lang spec + package docs + user query
└───────────┬─────────────┘
            │ full prompt
            ▼
┌─────────────────────────┐
│   Local LLM (Ollama)    │  generates BraneScript code
│   e.g. qwen3.5          │
└───────────┬─────────────┘
            │ raw LLM output
            ▼
┌─────────────────────────┐
│   Output cleanup        │  strips <think> tokens + markdown fences
└───────────┬─────────────┘
            │ cleaned code
            ▼
┌─────────────────────────┐
│   Validation loop       │  syntax check → semantic check
│   (up to 3 retries)     │  on failure: inject error into prompt, retry
└───────────┬─────────────┘
            │ valid BraneScript
            ▼
     saved to generated_branescripts/
```

### Two vector databases

| Database | Content | Purpose |
|---|---|---|
| `brane_lang_db` | BraneScript manual docs, spec, `.bs` examples, curated `(intent, code)` pairs, syntax reference | Teaches the LLM *how* to write BraneScript |
| `brane_pkg_db` | Package quick references, `container.yml`, configuration guides | Tells the LLM *what functions exist* in available packages |

---

## How the pipeline works step by step

### Step 1 — Intent decomposition (`intent_decomposer.py`)

The user's natural-language request is sent to the LLM with a structured prompt that asks it to break the intent into a numbered list of **concrete BraneScript sub-tasks** (e.g. "import healthcare package", "call analyze_heart_disease with patient JSON").

Each sub-task is then used as a search query against `brane_lang_db` to retrieve the most relevant **language spec chunks** (syntax rules, examples, API patterns). Duplicates are removed and a token budget caps the retrieved text so it fits within the LLM's context window.

### Step 2 — Package retrieval (`pkg_retriever.py`)

The full user query is used to search `brane_pkg_db` for the most relevant **package documentation** — function signatures, input/output types, and usage examples. This gives the LLM the concrete API it needs to call.

### Step 3 — Prompt assembly + generation (`pipeline.py`)

A structured prompt is assembled from:
- **Few-shot BraneScript examples** (hardcoded, always injected)
- **Absolute rules** (no Python, no markdown fences, use `:=`, etc.)
- The user request and sub-tasks
- The retrieved language spec context
- The retrieved package context

This prompt is sent to the local Ollama LLM. The output is then **cleaned** before any checks:

- `strip_thinking_tokens()` removes `<think>…</think>` blocks that Qwen3-family models may emit
- `strip_code_fences()` extracts code from markdown code blocks if the model wrapped it

### Step 4 — Validation with retry

Two checks run on the cleaned output:

1. **`looks_like_branescript()`** — basic structural check (presence of `:=`, `import`, `::`, etc.)
2. **`check_syntax()`** — heuristic checks for balanced braces/parentheses and correct `:=` assignment
3. **`check_semantic()`** — verifies that every `import` and `::` package reference appears in the retrieved package context (guards against hallucinated package names)

If any check fails and retries remain, the specific error is injected back into the prompt and the LLM is asked to correct it. Maximum 3 attempts.

### Step 5 — Save

The validated BraneScript is saved to `generated_branescripts/` with a timestamped filename derived from the user query.

---

## What changed and why

### Problem: the LLM was generating Python instead of BraneScript

The previous system consistently produced Python class hierarchies (`class HealthcareAnalyzer`, `def analyze_patient`, etc.) instead of BraneScript. There were three root causes:

#### 1. The knowledge base was broken and contained the wrong content

The `knowledgeBase.py` script had an undefined variable (`SELECTED_DOCS_PATH`) that caused the **entire manual documentation to be silently skipped**. The manual pages (`manual/src/branescript/*.md`) are the richest BraneScript teaching material — they contain natural-language explanations *with* annotated code examples — but they were never loaded.

What *was* loaded instead: the Brane compiler's Rust source files (`ast.rs`, `tokens.rs`, `bscript.rs`, etc.). These files are full of Rust-specific constructs and provide zero guidance on BraneScript syntax. The LLM would pattern-match on whatever it retrieved and naturally fell back to Python since that dominates its training data.

**Fix:** Removed all Rust source files from the language DB. Added:
- `submodules/manual/src/branescript/` — the full BraneScript manual chapter
- `submodules/manual/src/scientists/bscript/` — scientist-facing usage guide
- `submodules/specification/src/appendix/languages/bscript/` — language specification appendix
- `data/syntax_reference.md` — a compact, LLM-optimized BraneScript cheatsheet (new)
- `data/examples/*.jsonl` — 40 hand-curated `(intent, BraneScript)` pairs (new)

#### 2. The generation prompt had no examples

Without seeing a single BraneScript example, the LLM had no anchor for the output format. It would see the language name "BraneScript" and map it to whatever seemed closest in its training data.

**Fix:** A `BRANESCRIPT_FEW_SHOT` block is now injected at the top of every generation prompt. It shows three concrete examples: a package import + call, an if/else block, and a function definition.

#### 3. The LLM output was used raw

Qwen3-family models can emit `<think>...</think>` reasoning blocks before the actual answer. Even simpler models sometimes wrap output in markdown code fences (` ```python ... ``` `). Both of these would cause the downstream validation to fail.

**Fix:** Two cleanup functions now run immediately after the LLM responds, before any validation:
- `strip_thinking_tokens()` — removes `<think>…</think>` blocks
- `strip_code_fences()` — extracts the code content from any markdown fence

---

### New addition: example library + fine-tuning infrastructure

Even with a fixed knowledge base, a general-purpose LLM will always struggle with a niche language it has never truly learned. The long-term fix is **fine-tuning** — teaching the model BraneScript directly.

**`data/examples/`** — 40 curated `(intent, BraneScript)` pairs across 6 categories (basic, control flow, functions, classes/arrays, healthcare package, advanced). These are used both as RAG documents (retrieved when similar queries come in) and as training data.

**`src/example_generator.py`** — runs the local LLM against a set of seed intents to generate more examples automatically. You review and keep the good ones.

**`fine_tuning/`** — full fine-tuning pipeline using [unsloth](https://github.com/unslothai/unsloth) + QLoRA:
- `prepare_dataset.py` — converts examples to chat-format JSONL
- `train.py` — fine-tunes the model, exports to GGUF
- `Modelfile.template` — registers the model in Ollama

---

## Files in this folder

| File | Responsibility |
|---|---|
| `pipeline.py` | Main entry point. Orchestrates decomposition → retrieval → generation → validation → save |
| `intent_decomposer.py` | Breaks user intent into sub-tasks; retrieves language spec chunks from `brane_lang_db` |
| `pkg_retriever.py` | Retrieves package/dataset documentation from `brane_pkg_db` |
| `knowledgeBase.py` | Builds both vector databases from source documents; run once to (re)build |
| `example_generator.py` | Generates additional `(intent, BraneScript)` training examples using the local LLM |

---

## Quick start

### Prerequisites

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Make sure Ollama is running with your model
ollama serve &
ollama pull qwen3.5   # or whatever model you use

# 3. Build the knowledge bases (only needed once, or when docs change)
cd src
python knowledgeBase.py
```

### Run the pipeline

```bash
cd src
python pipeline.py
```

You will be prompted:
```
Enter the user request: Analyze heart disease risk for a 55-year-old male with blood pressure 150
```

The generated BraneScript is printed to the terminal and saved to `generated_branescripts/`.

---

## Running the pipeline

```bash
cd src
python pipeline.py
```

**Example inputs to try:**
- `"Analyze heart disease risk for a 47-year-old male patient with blood pressure 150 and heart rate 70"`
- `"Generate a health report for a 65-year-old female patient with diabetes history"`
- `"Run heart disease analysis in parallel for two patients and print both results"`
- `"Define a function that takes age and gender as parameters and returns a patient JSON string"`

**What you get back:** a `.brane` file in `generated_branescripts/` that can be submitted directly to a Brane instance with `brane run <file>`.

---

## Rebuilding the knowledge bases

Run this whenever you:
- Add new packages to `submodules/packages/`
- Add new examples to `data/examples/`
- Update `data/syntax_reference.md`

```bash
cd src
python knowledgeBase.py
```

This rebuilds both `brane_lang_db` (language spec) and `brane_pkg_db` (packages) from scratch.

---

## Generating more training examples

More examples → better retrieval → better output (and better fine-tuning).

```bash
# Generate 30 new examples using the local LLM
python src/example_generator.py --count 30

# Generate examples for a specific category only
python src/example_generator.py --category healthcare --count 10
```

Available categories: `basic`, `control_flow`, `functions`, `classes_arrays`, `healthcare`, `advanced`

Generated examples are saved to `data/examples/generated.jsonl`. **Review them before use** — the LLM may still produce incorrect output that you do not want in your training data.

After generating and reviewing:
```bash
# Rebuild the knowledge bases to include the new examples
python src/knowledgeBase.py
```

---

## Configuration reference

All tuneable constants are at the top of each file.

### `pipeline.py`

| Constant | Default | Description |
|---|---|---|
| `MAX_RETRIES` | `3` | Maximum generation+validation attempts before giving up |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for vector search |
| LLM `model` | `"qwen3.5"` | Ollama model name (change to `"branescript-qwen"` after fine-tuning) |
| LLM `temperature` | `0.6` | Lower = more deterministic; try `0.3`–`0.4` for fewer hallucinations |

### `intent_decomposer.py`

| Constant | Default | Description |
|---|---|---|
| `MAX_LANG_CONTEXT_CHARS` | `6000` | Token budget for retrieved language spec (increase if context window allows) |
| `k_per_subtask` | `3` | Number of DB chunks retrieved per sub-task |

### `pkg_retriever.py`

| Constant | Default | Description |
|---|---|---|
| `k` | `4` | Number of package doc chunks retrieved per query |
